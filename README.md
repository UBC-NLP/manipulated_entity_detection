# Automatic Detection of Entity-Manipulated Text using Factual Knowledge

Code and data used in our ACL 2022 paper titled [Automatic Detection of Entity-Manipulated Text using Factual Knowledge](https://arxiv.org/abs/2203.10343).

## Dependencies
* torch
* tensorflow
* transformers
* GPUtil
* torch_geometric
* sklearn

## Data
All the data used in the paper can be downloaded from [here](https://1drv.ms/u/s!AlflMXNPVy-wgpk_Oij5rYCXaLQWxQ?e=kpsLIq).

## Baseline Run
```
python roberta.py <data-path> <batch-size> <learning-rate>
```
where,
* `<data-path>`: path to the detection data (e.g., `data/gpt2_123/num_entities_3`)
* `<batch-size>`: batch size for fine-tuning (32)
* `<learning-rate>`: learning rate for fine-tuning (`1e-5`, `2e-5`, `3e-5`)

## Ours Run
```
python geometric_baseline.py <data-path> <batch-size> <learning-rate> <gnn-type> <wiki-init> <gnn-num-feats> <mp-iter> <first-hop-neighbors-path> <entity-supervision>
```
where,
* `<data-path>`: path to the detection data (e.g., `data/gpt2_123/num_entities_3`)
* `<batch-size>`: batch size for fine-tuning (32)
* `<learning-rate>`: learning rate for fine-tuning (`1e-5`, `2e-5`, `3e-5`)
* `<gnn-type>`: type of GNN (e.g., `GCNConv`)
* `<wiki-init>`: need to initialize node embeddings with wikipedia2vec embeddings? (0 or 1) (ensure `WIKI_EMB_F` in the code is set to path of the wikipedia2vec embeddings)
* `<gnn-num-feats>`: hidden dimension of the GNN (300)
* `<mp-iter>`: number of message passing iterations (aka depth of GNN) (1, 2, 3)
* `<first-hop-neighbors-path>`: path to the first hop neighbors of each node (e.g., `data/kb_first_hop_neighbors_entity_replace/nonlm_123_leastfreq/num_entities_3`)
* `<entity-supervision>`: need entity supervision? (0 or 1)

## Cite
If you find this project useful, please cite the paper:
```
@inproceedings{jawahar-etal-2022-automatic,
    title = "Automatic Detection of Entity-Manipulated Text using Factual Knowledge",
    author = "Jawahar, Ganesh  and
      Abdul-Mageed, Muhammad  and
      Lakshmanan, Laks",
    booktitle = "Association for Computational Linguistics",
    year = "2022",
    pages = "86--93"
}
```

## Contact
If you have any questions or suggestions, please contact [Ganesh Jawahar](mailto:ganeshjwhr@gmail.com).

## License
This repository is GPL-licensed.

