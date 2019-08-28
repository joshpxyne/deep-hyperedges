# Deep Hyperedges

This project implements the hypergraph learning techniques introduced in ____.

### HypergraphWalks.py
This file implements the SubsampleAndTraverse and TraverseAndSelect random walk procedures, with constant or inverse switching probabilities. 

### Models.py
This file implements the three models discussed: DeepHyperedges, which combines knowledge from both random walk procedures; MLP, which is a neural network used to classify the hyperedge embeddings from TraverseAndSelect; and DeepSets, which is used to classify sets of vertex embeddings from SubsampleAndTraverse.

### Embeddings.py
This file implements Word2Vec using gensim to create embeddings for the random walks.

Each of the Jupyter notebooks implement the experiments performed for the five datasets tested on. For each dataset, create the following file structure:

```
logs
|__<dataset name>
   |__deephyperedges_logs
   |__deepsets_logs
   |__MLP_logs

weights
|__<dataset name>

data
|__<dataset name> e.g., cora, corum, disgenet, meetups, pubmed. These names are used in the notebooks.
```
In the `data` directory, put the appropriate data from the following sources:
- Cora CS publication citation network dataset: https://relational.fit.cvut.cz/dataset/CORA
- CORUM protein complex dataset: https://mips.helmholtz-muenchen.de/corum/#download
- DisGeNet disease genomics dataset: http://www.disgenet.org/downloads
- Meetups social networking dataset: https://www.kaggle.com/sirpunch/meetups-data-from-meetupcom
- PubMed diabetes publication citation network dataset: https://linqs.soe.ucsc.edu/data
