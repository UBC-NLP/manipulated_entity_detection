'''
roberta model
'''

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm import tqdm, trange
import numpy as np
import io
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import sys
import json
import GPUtil

## Set seed of randomization and working device
manual_seed = 77
torch.manual_seed(manual_seed)
if torch.cuda.is_available():
  device_ids = GPUtil.getAvailable(limit = 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
  torch.cuda.manual_seed(manual_seed)

data_dir = sys.argv[1]

# define a function for data preparation
def data_prepare(split, tokenizer, max_len = 256):
  # read from dataset
  input_ids, labels = [], []
  for li, line in enumerate(open(data_dir + "/" + split + ".jsonl")):
    content = json.loads(line.strip())
    # tokenize text
    tokenized_text = tokenizer.tokenize(content["text"]) # "[CLS] " + 
    tokenized_text = tokenized_text[:max_len+1] # + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    input_ids.append(token_ids)
    # label
    labels.append(content["label"])
    # if li > 50:
    #   break

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

  return inputs, labels, masks

tokenizer = RobertaTokenizer.from_pretrained("roberta-large", cache_dir="/tmp/cache_dir") 
train_inputs, train_labels, train_masks = data_prepare('train', tokenizer)
validation_inputs, validation_labels, validation_masks = data_prepare('val', tokenizer)
test_inputs, test_labels, test_masks = data_prepare('test', tokenizer)

class RoBERTaDetector(nn.Module):
  def __init__(self, model_path, hidden_size):
    super(RoBERTaDetector, self).__init__()
    self.model_path = model_path
    self.hidden_size = hidden_size
    self.bert_model = RobertaModel.from_pretrained("roberta-large", cache_dir=model_path, output_hidden_states=True, output_attentions=True) #, cache_dir="/scratch/ganeshjw/objects/manipulated_detection/slurm_outs/cache_dir")
    self.label_num = 2
    self.fc = nn.Linear(self.hidden_size, self.label_num)

  def forward(self, bert_ids, bert_mask):
    a, pooler_output, b, c = self.bert_model(input_ids=bert_ids, attention_mask=bert_mask)
    fc_output = self.fc(pooler_output)
    return fc_output

def train(model, iterator, optimizer, scheduler, criterion):
  model.train()
  epoch_loss = 0
  pbar = tqdm(total=1+(train_inputs.shape[0]//batch_size))
  for i, batch in enumerate(iterator):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    input_ids, input_mask, labels = batch
    
    optimizer.zero_grad()
    outputs = model(input_ids, input_mask)

    loss = criterion(outputs, labels)
    # delete used variables to free GPU memory
    del batch, input_ids, input_mask, labels
    #loss.backward()
    
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

  pbar.close()
  
  # free GPU memory
  if device == 'cuda':
    torch.cuda.empty_cache()

  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
  model.eval()
  epoch_loss = 0
  all_pred=[]
  all_label = []
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
      input_ids, input_mask, labels = batch

      outputs = model(input_ids, input_mask)
      
      loss = criterion(outputs, labels)

      # delete used variables to free GPU memory
      del batch, input_ids, input_mask
      epoch_loss += loss.cpu().item()

      # identify the predicted class for each example in the batch
      probabilities, predicted = torch.max(outputs.cpu().data, 1)

      # put all the true labels and predictions to two lists
      all_pred.extend(predicted)
      all_label.extend(labels.cpu())
  
  acc = 0.0
  for lab, pred in zip(all_label, all_pred):
    if lab.item() == pred.item():
      acc += 1.0
  accuracy = acc / float(len(all_label))
  return epoch_loss / len(iterator), accuracy 

def writestat(model, iterator, criterion):
  model.eval()
  epoch_loss = 0
  all_pred=[]
  all_label = []
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
      input_ids, input_mask, labels = batch

      outputs = model(input_ids, input_mask)
      
      loss = criterion(outputs, labels)

      # delete used variables to free GPU memory
      del batch, input_ids, input_mask
      epoch_loss += loss.cpu().item()

      # identify the predicted class for each example in the batch
      probabilities, predicted = torch.max(outputs.cpu().data, 1)

      # put all the true labels and predictions to two lists
      all_pred.extend(predicted)
      all_label.extend(labels.cpu())
  
  w = open(write_stat_f, "w")
  i = 0
  for lab, pred in zip(all_label, all_pred):
    score = 1 if lab.item() == pred.item() else 0
    w.write("pred-statsigni:%d=%d\n"%(i, score))
    #print("pred-statsigni:%d=%d"%(i, score))
    i += 1
  #  w.write(str(score)+"\n")
  w.close()

# Train the model
# Parameters:
lr = float(sys.argv[3]) # 2e-5 # 1e-5, 2e-5, 3e-5
max_grad_norm = 1.0
epochs = 10 # 10 epochs
warmup_proportion = 0.1
batch_size = int(sys.argv[2]) #32
hidden_size = 1024 #768
write_stat_f = sys.argv[4]

# We'll take training samples in random order in each epoch. 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, sampler = RandomSampler(train_data), batch_size=batch_size)

# We'll just read validation set sequentially.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(validation_data, 
                                   sampler = SequentialSampler(validation_data), # Pull out batches sequentially.
                                   batch_size=batch_size)

# We'll just read test set sequentially.
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, 
                                   sampler = SequentialSampler(test_data), # Pull out batches sequentially.
                                   batch_size=batch_size)

bert_model = RoBERTaDetector("/shared/objects/cache_dir", hidden_size).to(device)
if torch.cuda.is_available():
  if torch.cuda.device_count() == 1:
    bert_model = bert_model.to(device)
  else:
    print("more gpus")
    device_ids = GPUtil.getAvailable(limit = 4)
    torch.backends.cudnn.benchmark = True
    bert_model = bert_model.to(device)
    bert_model = nn.DataParallel(bert_model, device_ids=device_ids)

### In Transformers, optimizer and schedules are instantiated like this:
# Note: AdamW is a class from the huggingface library
# the 'W' stands for 'Weight Decay"
optimizer = AdamW(bert_model.parameters(), lr=lr, correct_bias=False)
# schedules
num_training_steps  = len(train_dataloader) * epochs
num_warmup_steps = num_training_steps * warmup_proportion
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

# We use nn.CrossEntropyLoss() as our loss function. 
criterion = nn.CrossEntropyLoss()
prev_val_acc, best_val_acc, best_test_acc, best_epoch, patience = None, None, None, 0, 0
for epoch in range(epochs):
  train_loss = train(bert_model, train_dataloader, optimizer, scheduler, criterion)  
  val_loss, val_acc = evaluate(bert_model, validation_dataloader, criterion)
  if not best_val_acc or val_acc > best_val_acc:
    best_val_acc = val_acc
    _, best_test_acc = evaluate(bert_model, test_dataloader, criterion)
    final_epoch = epoch + 1
    writestat(bert_model, test_dataloader, criterion)
  print('\n Epoch [{}/{}], Train Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation Loss: {:.4f}, Best Test Accuracy: {:.4f}'.format(epoch+1, epochs, train_loss, val_acc, val_loss, best_test_acc))
  prev_val_acc = val_acc
print('\n {} {} {} Best_epoch [{}/{}], Best Validation Accuracy: {:.4f}, Best Test Accuracy: {:.4f}'.format(sys.argv[1], sys.argv[2], sys.argv[3], final_epoch, epochs, best_val_acc, best_test_acc))


