# Deep Hyperedges
This repository implements the hypergraph learning techniques introduced in [Deep Hyperedges: A Framework for Transductive and Inductive Learning on Hypergraphs](https://arxiv.org/abs/1910.02633), to be presented in December at the Sets & Partitions Workshop at the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019). This is a very important and exciting problem, so we're encouraging collaboration here. Work on this particular framework is still very much in an active state.

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
- Cora CS publication citation network dataset: https://relational.fit.cvut.cz/dataset/CORA. Vertices are papers and hyperedges are the cited works of a given paper (so its 1-neighborhood). A bit contrived, I know, but we do see a performance increase over the graph structure by itself. Each paper (and thus each hyperedge) is classified into one of seven classes based on topic.
- CORUM protein complex dataset: https://mips.helmholtz-muenchen.de/corum/#download. Vertices are proteins and hyperedges are collections of proteins. Each hyperedge is labeled based on whether or not the collection forms a protein complex. Negative examples are generated in the notebook.
- DisGeNet disease genomics dataset: http://www.disgenet.org/downloads. Vertices are genes and hyperedges are diseases. Each disease is classified into one of 23 MeSH codes (if it has multiple, we randomly select one).
- Meetups social networking dataset: https://www.kaggle.com/sirpunch/meetups-data-from-meetupcom. Vertices are members and hyperedges are meetup events. Each meetup event is classified into one of 36 types. We also experiment with combining the two largest--"Tech" and "Career & Business"--into one class and all others into another class to give a balanced dataset.
- PubMed diabetes publication citation network dataset: https://linqs.soe.ucsc.edu/data. Vertices are papers and hyperedges are the cited works of a given paper. Again, a bit contrived, but there are more baselines to compare with since there's an underlying (directed) graph structure and a large body of work which studies deep learning on graphs. Each paper (and thus each hyperedge) is classified into one of three classes based on the type of diabetes it studies.

Citation:
```
@misc{payne2019deep,
    title={Deep Hyperedges: a Framework for Transductive and Inductive Learning on Hypergraphs},
    author={Josh Payne},
    year={2019},
    eprint={1910.02633},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

