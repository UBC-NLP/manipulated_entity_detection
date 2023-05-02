'''
roberta + entity graph
'''

import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch_geometric.nn import GCNConv #, GATConv
from tqdm import tqdm, trange
import numpy as np
import io
import os
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
import sys
import json
import GPUtil
from collections import Counter
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

## Set seed of randomization and working device
manual_seed = 77
torch.manual_seed(manual_seed)
if torch.cuda.is_available():
  device_ids = GPUtil.getAvailable(limit = 4)
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
  torch.cuda.manual_seed(manual_seed)

trial_run = False
data_dir = sys.argv[1]
kb_edges_dir = sys.argv[8]

# Parameters:
batch_size = int(sys.argv[2]) if not trial_run else 4 #32
lr = float(sys.argv[3]) if not trial_run else 2e-5 # 2e-5 # 1e-5, 3e-5
max_grad_norm = 1.0
epochs = 10 if not trial_run else 1 # 10 epochs
warmup_proportion = 0.1
hidden_size = 1024 if not trial_run else 768 # ???
gnn_type = sys.argv[4] if not trial_run else "GCNConv"
NODE_FREQ = 10
GNN_NUM_FEATS = int(sys.argv[6]) if not trial_run else 15
GNN_MP_ITERATIONS = int(sys.argv[7]) if not trial_run else 2 # should be at least 1
WIKI_INIT = int(sys.argv[5]) if not trial_run else 0
WIKI_EMB_F = "/shared/objects/manipulated_detection/data/wikipedia2vec/enwiki_20180420_%dd.txt"%GNN_NUM_FEATS
ENTITY_SUPERVISION = int(sys.argv[9]) if not trial_run else 1
ROBERTA_ENTITY_INIT = int(sys.argv[10]) if not trial_run else 1
CACHE_DIR = "/shared/objects/cache_dir"
trial_run = False
WRITE_STATE_F = sys.argv[12]

def read_all(f, key, val):
  text2info = {}
  for line in open(f):
    content = json.loads(line.strip())
    text2info[content[key]] = content[val]
  return text2info

def localize_nodes_edges(global_nodes, global_edges):
  # create global to local node maps
  glob2loc, loc2glob = {}, {}
  for loc, glo in enumerate(global_nodes):
    glob2loc[glo] = loc
    loc2glob[loc] = glo
  # fill local nodes and local edges
  local_nodes, local_edges = [], []
  for glo in global_nodes:
    local_nodes.append(glob2loc[glo])
  for edge in global_edges:
    local_edges.append([glob2loc[edge[0]], glob2loc[edge[1]]])
  return local_nodes, local_edges, glob2loc, loc2glob

# define a function for data preparation
UNK_TOKEN_S = '<UNK-S>'
UNK_TOKEN_R = '<UNK-R>'
UNK_TOKEN_O = '<UNK-O>'
NULL_GRAPH = '<NULL-G>'
node2id, id2node, node2freq = {UNK_TOKEN_S: 0, UNK_TOKEN_R: 1, UNK_TOKEN_O: 2, NULL_GRAPH: 3}, {0: UNK_TOKEN_S, 1: UNK_TOKEN_R, 2: UNK_TOKEN_O, 3: NULL_GRAPH}, Counter()
def data_prepare(split, tokenizer, max_len = 256):
  fake_text2edges = read_all(kb_edges_dir + "/" + split + ".jsonl", "text", "edges")
  real_text2edges = read_all(kb_edges_dir + "/human." + split + ".jsonl", "text", "edges")
  # read from dataset
  input_ids, labels, raw_edges, manip_entities, entity2tokenids = [], [], [], [], []
  for li, line in enumerate(open(data_dir + "/" + split + ".jsonl")):
    content = json.loads(line.strip())
    edges = fake_text2edges[content["text"]] if content["label"] == 0 else real_text2edges[content["text"]]
    # tokenize text
    tokenized_text = tokenizer.tokenize("[CLS] " + content["text"])
    tokenized_text = tokenized_text[:max_len+1] + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    input_ids.append(token_ids)
    # label
    labels.append(content["label"])
    # edges
    if split == 'train':
      for entity in edges:
        for triplet in edges[entity]:
          node2freq[triplet[0]] += 1
          node2freq[triplet[1]] += 1
          node2freq[triplet[2]] += 1
    raw_edges.append(edges)
    # get manipulated entities
    if "replacements" not in content:
      assert(content["label"] == 1)
      manip_entities.append(None)
    else:
      entities = {}
      for replacement in content["replacements"]:
        new_entity = replacement["new_entity"].lower().replace(" ", "_") if type(replacement) is not list else replacement[1].lower().replace(" ", "_")
        entities[new_entity] = True
        for entity in edges:
          for triplet in edges[entity]:
            if triplet[0] in new_entity or new_entity in triplet[0]:
              entities[triplet[0]] = True
            if triplet[1] in new_entity or new_entity in triplet[1]:
              entities[triplet[1]] = True
            if triplet[2] in new_entity or new_entity in triplet[2]:
              entities[triplet[2]] = True
      manip_entities.append(list(entities))
    if trial_run and li > 50:
      break

  # prune nodes based on freq and create node maps
  if split == 'train':
    for node in node2freq:
      if node2freq[node] >= NODE_FREQ:
        node2id[node] = len(id2node)
        id2node[node2id[node]] = node

  # create nodes and edges tensors
  graph_nodes, graph_edges, graph_doc_ents, graph_manip_ents = [], [], [], []
  num_cases = 0
  for inst_i, edges in enumerate(raw_edges):
    cur_nodes, cur_edges, doc_entities = {}, {}, {}
    for entity in edges:
      for edge in edges[entity]:
        # add nodes
        for node, typ in [(edge[0], 'S'), (edge[1], 'R'), (edge[2], 'O')]:
          if node in node2id:
            cur_nodes[node2id[node]] = True
          else:
            cur_nodes[node2id['<UNK-%s>'%typ]] = True
        # add edges
        # remove edges which are all UNKs
        if edge[0] not in node2id and edge[1] not in node2id and edge[2] not in node2id:
          continue
        cur_edges[node2id[edge[0]] if edge[0] in node2id else node2id['<UNK-S>'], node2id[edge[1]] if edge[1] in node2id else node2id['<UNK-R>']] = True
        cur_edges[node2id[edge[1]] if edge[1] in node2id else node2id['<UNK-R>'], node2id[edge[2]] if edge[2] in node2id else node2id['<UNK-O>']] = True
      if entity in node2id:
        doc_entities[node2id[entity]] = True
    # handle empty graph
    cur_nodes = list(cur_nodes)
    if len(cur_nodes) == 0:
      assert(len(cur_edges) == 0)
      cur_nodes.append(node2id[NULL_GRAPH])
      doc_entities[node2id[NULL_GRAPH]] = True
    elif len(doc_entities) == 0:
      doc_entities[node2id['<UNK-S>']] = True
      # cur_edges[node2id[NULL_GRAPH], node2id[NULL_GRAPH]] = True
    cur_edges = list(cur_edges)
    cur_nodes, cur_edges, glob2loc, loc2glob = localize_nodes_edges(cur_nodes, cur_edges)
    # transpose edges
    cur_edges = np.array(cur_edges).transpose().tolist()
    graph_nodes.append(torch.tensor(cur_nodes, dtype=torch.long))
    graph_edges.append(torch.tensor(cur_edges, dtype=torch.long))
    graph_doc_ents.append([[glob2loc[entity] for entity in doc_entities], len(cur_nodes)])
    # manip ents
    cur_manip_ent_labels = []
    for node in cur_nodes:
      if not manip_entities[inst_i]:
        cur_manip_ent_labels.append(0)
      elif id2node[loc2glob[node]] in manip_entities[inst_i]:
        cur_manip_ent_labels.append(1)
        #print(cur_manip_ent_labels)
        num_cases += 1
      else:
        cur_manip_ent_labels.append(0)
    graph_manip_ents.append(torch.tensor(cur_manip_ent_labels, dtype=torch.long))
  if trial_run:
    print(f"# cases = {num_cases}")

  # pad our input seqeunce to the fixed length (i.e., max_len) with index of [PAD] token
  pad_ind = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
  input_ids = pad_sequences(input_ids, maxlen=max_len+2, dtype="long", truncating="post", padding="post", value=pad_ind)

  # create attention masks
  attention_masks = []
  # create a mask of 1s for each token followed by 0s for pad tokens
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

  # Convert all of our data into torch tensors, the required datatype for our model
  inputs = torch.tensor(input_ids)
  labels = torch.tensor(labels)
  masks = torch.tensor(attention_masks)

  dataset = []
  for it in range(inputs.shape[0]):
    if trial_run and it < 2:
      print(graph_nodes[it])
      print(graph_edges[it])
    data = Data(inputs=inputs[it].unsqueeze(0), labels=labels[it].unsqueeze(0), masks=masks[it].unsqueeze(0), graph_nodes=graph_nodes[it], edge_index=graph_edges[it], graph_doc_ents=graph_doc_ents[it], graph_manip_ents=graph_manip_ents[it])
    data.num_nodes = len(graph_nodes[it])
    dataset.append(data)
  return dataset #inputs, labels, masks, graph_nodes, graph_edges

# Train the model
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir=CACHE_DIR) #'roberta-large' if not trial_run else 'roberta-base', cache_dir=CACHE_DIR)
train_dataset = data_prepare('train', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True if not trial_run else False)
for batch in train_dataloader:
  if trial_run:
    print(batch.graph_nodes)
    print(batch.edge_index)
    print(batch.inputs.size())
    print(batch.masks.size())
    print(batch.graph_doc_ents)
    break

validation_dataset = data_prepare('val', tokenizer)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

test_dataset = data_prepare('test', tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class RoBERTaDetector(nn.Module):
  def __init__(self, model_path, hidden_size):
    super(RoBERTaDetector, self).__init__()
    self.model_path = model_path
    self.hidden_size = hidden_size
    self.bert_model = RobertaModel.from_pretrained('roberta-large', cache_dir=CACHE_DIR, output_hidden_states=True, output_attentions=True).to(device) # , cache_dir=CACHE_DIR , return_dict=False
    self.label_num = 2
    self.num_features = GNN_NUM_FEATS
    self.num_mp_iterations = GNN_MP_ITERATIONS
    self.dropout = nn.Dropout(p=0.2).to(device)
    self.fc = nn.Linear(self.hidden_size if gnn_type == "None" else self.hidden_size + self.num_features, self.label_num).to(device)
    if gnn_type != "None":
      self.nodeEmb = nn.Embedding(len(node2id), self.num_features)
      self.wiki_init()
      self.nodeEmb = self.nodeEmb.to(device)
      if gnn_type == "GCNConv":
        self.gnn_layers = [GCNConv(self.num_features, self.num_features).to(device)] 
        for lay_id in range(self.num_mp_iterations-1):
          self.gnn_layers.append(GCNConv(self.num_features, self.num_features).to(device))
      elif gnn_type == "GATConv":
        self.gnn_layers = [GATConv(self.num_features, self.num_features).to(device)] 
        for lay_id in range(self.num_mp_iterations-1):
          self.gnn_layers.append(GATConv(self.num_features, self.num_features).to(device))
      #self.graph_out_layer = nn.Linear(self.num_features, self.num_features).to(device)
      #self.graph_drop_layer = nn.Dropout(p=0.1).to(device)
      if ENTITY_SUPERVISION == 1:
        self.entity_classifier = nn.Linear(self.num_features, 2).to(device)

  def wiki_init(self):
    if WIKI_INIT == 1:
      num_hits = 0
      for li, line in enumerate(open(WIKI_EMB_F)):
        if li == 0:
          continue
        content = line.strip().split()
        entity_name = ''.join(content[0:-300])
        if entity_name.startswith("ENTITY/"):
          if entity_name.split("ENTITY/")[-1].lower() in node2id:
            self.nodeEmb.weight.data[node2id[entity_name.split("ENTITY/")[-1].lower()]] = torch.tensor([float(dim) for dim in content[-self.num_features:]], dtype=torch.float)
            num_hits += 1
      print('#nodes initialized from wikipedia2vec = %d (%.2f)'%(num_hits, float(num_hits)/float(len(node2id))))

  def forward(self, bert_ids, bert_mask, graph_nodes=None, edge_index=None, graph_doc_ents=None, graph_manip_ents=None):
    _, pooler_output, _, _ = self.bert_model(input_ids=bert_ids.to(device), attention_mask=bert_mask.to(device))
    entity_emb = self.nodeEmb(graph_nodes.to(device)) # num_nodes_in_batch x num_features
    # message passing
    message = self.gnn_layers[0](entity_emb, edge_index.to(device))
    message = message.tanh()
    for lay_id in range(self.num_mp_iterations-1):
      message = self.gnn_layers[lay_id+1](message, edge_index.to(device))
      message = message.tanh()
    entity_out, start_idx = [], 0
    for inst_i in range(len(graph_doc_ents)):
      # mean of all document embeddings
      entity_out.append(torch.index_select(message, 0, torch.tensor([start_idx + item for item in graph_doc_ents[inst_i][0]], dtype=torch.long).to(device)).mean(0))
      start_idx += graph_doc_ents[inst_i][1]
    entity_out = torch.stack(entity_out)
    # entity_out = self.graph_drop_layer(nn.ReLU()(self.graph_out_layer(entity_out)))
    # merge pooler + entity out
    merged_input = self.dropout(nn.ReLU()(torch.cat([pooler_output, entity_out], dim=1)))
    #print(merged_input.size(), pooler_output.size(), entity_out.size())
    fc_output = self.fc(merged_input)
    if ENTITY_SUPERVISION == 1:
      return fc_output, self.dropout(nn.ReLU()(self.entity_classifier(message)))
    else:
      return fc_output

def train(model, iterator, optimizer, scheduler, criterion):
  model.train()
  epoch_loss = 0
  pbar = tqdm(total=1+(len(iterator)//batch_size))
  for i, batch in enumerate(iterator):
    # Add batch to GPU
    # batch = tuple(t.to(device) for t in batch)
    batch.inputs.to(device); batch.labels.to(device); batch.masks.to(device); batch.graph_nodes.to(device); batch.edge_index.to(device); batch.graph_manip_ents.to(device)

    # Unpack the inputs from our dataloader
    # input_ids, input_mask, labels = batch
    
    optimizer.zero_grad()
    outputs = None
    if gnn_type == "None":
      outputs = model(batch.inputs, batch.masks)
      loss = criterion(outputs, batch.labels.to(device))
    else:
      if ENTITY_SUPERVISION != 1:
        outputs = model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
        loss = criterion(outputs, batch.labels.to(device))
      else:
        outputs, ent_outputs = model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
        loss = criterion(outputs, batch.labels.to(device))
        loss += criterion(ent_outputs, batch.graph_manip_ents.to(device))

    # delete used variables to free GPU memory
    del batch
    
    # Backward pass
    if torch.cuda.device_count() == 1:
      loss.backward()
      epoch_loss += loss.cpu().item()
    else:
      loss.sum().backward()
      epoch_loss += loss.cpu().sum().item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore

    optimizer.step()
    scheduler.step()

    pbar.update(1)

    if trial_run:
      break

  pbar.close()
  
  # free GPU memory
  if device == 'cuda':
    torch.cuda.empty_cache()

  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
  model.eval()
  epoch_loss = 0
  all_pred = []
  all_label = []
  if ENTITY_SUPERVISION == 1:
    man_entity_pred = []
    man_entity_label = []
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      # Add batch to GPU
      # batch = tuple(t.to(device) for t in batch)
      batch.inputs.to(device); batch.labels.to(device); batch.masks.to(device); batch.graph_nodes.to(device); batch.edge_index.to(device); batch.graph_manip_ents.to(device);

      # Unpack the inputs from our dataloader
      # input_ids, input_mask, labels = batch

      #outputs = model(input_ids, input_mask)
      #outputs = model(batch.inputs, batch.masks) if gnn_type == "None" else model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
      outputs = None
      if gnn_type == "None":
        outputs = model(batch.inputs, batch.masks)
        loss = criterion(outputs, batch.labels.to(device))
      else:
        if ENTITY_SUPERVISION != 1:
          outputs = model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
          loss = criterion(outputs, batch.labels.to(device))
        else:
          outputs, ent_outputs = model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
          loss = criterion(outputs, batch.labels.to(device))
          loss += criterion(ent_outputs, batch.graph_manip_ents.to(device))
          probabilities, predicted = torch.max(ent_outputs.cpu().data, 1)
          man_entity_pred.extend(predicted)
          man_entity_label.extend(batch.graph_manip_ents.cpu())

      # loss = criterion(outputs, batch.labels.to(device))
      all_label.extend(batch.labels.cpu())

      # delete used variables to free GPU memory
      del batch #, input_ids, input_mask
      epoch_loss += loss.cpu().item()

      # identify the predicted class for each example in the batch
      probabilities, predicted = torch.max(outputs.cpu().data, 1)

      # put all the true labels and predictions to two lists
      all_pred.extend(predicted)

      if trial_run:
        break
  
  acc = 0.0
  for lab, pred in zip(all_label, all_pred):
    if lab.item() == pred.item():
      acc += 1.0
  accuracy = acc / float(len(all_label))
  if ENTITY_SUPERVISION != 1:
    return epoch_loss / len(iterator), accuracy 
  else:
    acc = 0.0
    num_pred_ones, num_gold_ones = 0, 0 # for recall
    for lab, pred in zip(man_entity_label, man_entity_pred):
      if lab.item() == pred.item():
        acc += 1.0
      if lab.item() == 1:
        num_gold_ones += 1.0
        if pred.item() == 1:
          num_pred_ones += 1.0
    man_accuracy = acc / float(len(man_entity_pred))
    man_ent_recall = num_pred_ones / num_gold_ones if num_gold_ones!=0 else 0
    prec, rec, fsc = precision_score(man_entity_label, man_entity_pred, average='macro'), recall_score(man_entity_label, man_entity_pred, average='macro'), f1_score(man_entity_label, man_entity_pred, average='macro')
    return epoch_loss / len(iterator), accuracy, man_accuracy, man_ent_recall, prec, rec, fsc

def writeres(model, iterator, criterion):
  model.eval()
  epoch_loss = 0
  all_pred = []
  all_label = []
  if ENTITY_SUPERVISION == 1:
    man_entity_pred = []
    man_entity_label = []
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      # Add batch to GPU
      # batch = tuple(t.to(device) for t in batch)
      batch.inputs.to(device); batch.labels.to(device); batch.masks.to(device); batch.graph_nodes.to(device); batch.edge_index.to(device); batch.graph_manip_ents.to(device);

      # Unpack the inputs from our dataloader
      # input_ids, input_mask, labels = batch

      #outputs = model(input_ids, input_mask)
      #outputs = model(batch.inputs, batch.masks) if gnn_type == "None" else model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
      outputs = None
      if gnn_type == "None":
        outputs = model(batch.inputs, batch.masks)
        loss = criterion(outputs, batch.labels.to(device))
      else:
        if ENTITY_SUPERVISION != 1:
          outputs = model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
          loss = criterion(outputs, batch.labels.to(device))
        else:
          outputs, ent_outputs = model(batch.inputs, batch.masks, graph_nodes=batch.graph_nodes, edge_index=batch.edge_index, graph_doc_ents=batch.graph_doc_ents)
          loss = criterion(outputs, batch.labels.to(device))
          loss += criterion(ent_outputs, batch.graph_manip_ents.to(device))
          probabilities, predicted = torch.max(ent_outputs.cpu().data, 1)
          man_entity_pred.extend(predicted)
          man_entity_label.extend(batch.graph_manip_ents.cpu())

      # loss = criterion(outputs, batch.labels.to(device))
      all_label.extend(batch.labels.cpu())

      # delete used variables to free GPU memory
      del batch #, input_ids, input_mask
      epoch_loss += loss.cpu().item()

      # identify the predicted class for each example in the batch
      probabilities, predicted = torch.max(outputs.cpu().data, 1)

      # put all the true labels and predictions to two lists
      all_pred.extend(predicted)

      if trial_run:
        break
  
  to_scores = []
  for lab, pred in zip(all_label, all_pred):
    to_scores.append(1 if lab.item() == pred.item() else 0)
  #w = open(WRITE_F, "w")
  #for score in to_scores:
  #  w.write(str(score)+"\n")
  #w.close()
  w = open(WRITE_STATE_F, "w")
  for score in to_scores:
    #w.write(str(score)+"\n")
    w.write("pred-statsigni:%d=%d\n"%(i, score))
  w.close()

bert_model = RoBERTaDetector('roberta-large' if not trial_run else 'roberta-base', hidden_size).to(device)
if torch.cuda.is_available():
  if torch.cuda.device_count() == 1:
    bert_model = bert_model.to(device)
  else:
    print("more gpus")
    device_ids = GPUtil.getAvailable(limit = 4)
    torch.backends.cudnn.benchmark = True
    bert_model = bert_model.to(device)
    bert_model = nn.DataParallel(bert_model, device_ids=device_ids)

optimizer = AdamW(bert_model.parameters(), lr=lr, correct_bias=False)
# schedules
if trial_run:
  print(f"num_training_steps per epoch = {len(train_dataloader)}")
num_training_steps  = len(train_dataloader) * epochs
num_warmup_steps = num_training_steps * warmup_proportion
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

# We use nn.CrossEntropyLoss() as our loss function. 
criterion = nn.CrossEntropyLoss()
prev_val_acc, best_val_acc, best_test_acc, best_epoch, patience = None, None, None, 0, 0
if ENTITY_SUPERVISION == 1:
  best_ent_acc, best_man_ent_recall = None, None
  best_ent_prec, best_ent_rec, best_ent_fsc = None, None, None
for epoch in range(epochs):
  train_loss = train(bert_model, train_dataloader, optimizer, scheduler, criterion)
  if ENTITY_SUPERVISION != 1:
    val_loss, val_acc = evaluate(bert_model, validation_dataloader, criterion)
    if not best_val_acc or val_acc > best_val_acc:
      best_val_acc = val_acc
      _, best_test_acc = evaluate(bert_model, test_dataloader, criterion)
      final_epoch = epoch + 1
      writeres(bert_model, test_dataloader, criterion)
    print('\n Epoch [{}/{}], Train Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation Loss: {:.4f}, Best Test Accuracy: {:.4f}'.format(epoch+1, epochs, train_loss, val_acc, val_loss, best_test_acc))
  else:
    val_loss, val_acc, man_accuracy, man_ent_recall, man_ent_p, man_ent_r, man_ent_f = evaluate(bert_model, validation_dataloader, criterion)
    if not best_val_acc or val_acc > best_val_acc:
      best_val_acc = val_acc
      _, best_test_acc, best_ent_acc, best_man_ent_recall, best_ent_prec, best_ent_rec, best_ent_fsc = evaluate(bert_model, test_dataloader, criterion)
      final_epoch = epoch + 1
      writeres(bert_model, test_dataloader, criterion)
    print('\n Epoch [{}/{}], Train Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation Loss: {:.4f}, Best Test Accuracy: {:.4f}, Best Man. Ent. Accuracy: {:.4f}, Best Man. Ent. Recall: {:.4f} {:.4f} {:.4f} {:.4f}'.format(epoch+1, epochs, train_loss, val_acc, val_loss, best_test_acc, man_accuracy, man_ent_recall, man_ent_p, man_ent_r, man_ent_f))
command_line_args_str = '_'.join(sys.argv[1:]) if not trial_run else ''
if ENTITY_SUPERVISION != 1:
  print('\n {} Best_epoch [{}/{}], Best Validation Accuracy: {:.4f}, Best Test Accuracy: {:.4f}'.format(command_line_args_str, final_epoch, epochs, best_val_acc, best_test_acc))
else:
  print('\n {} Best_epoch [{}/{}], Best Validation Accuracy: {:.4f}, Best Test Accuracy: {:.4f}, Best Man. Ent. Accuracy: {:.4f}, Best Man. Ent. Recall: {:.4f} {:.4f} {:.4f} {:.4f}'.format(command_line_args_str, final_epoch, epochs, best_val_acc, best_test_acc, man_accuracy, man_ent_recall, best_ent_prec, best_ent_rec, best_ent_fsc))





