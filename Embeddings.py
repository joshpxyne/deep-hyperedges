from gensim.models import Word2Vec
from gensim.models.poincare import PoincareModel, PoincareRelations
import time as time

def EmbedWord2Vec(walks,dimension):
    time_start = time.time()
    print("Creating embeddings.")
    model = Word2Vec(walks, size=dimension, window=5, min_count=0, sg=1, workers=32, iter=1)
    node_ids = model.wv.index2word
    node_embeddings = model.wv.vectors
    print("Embedding generation runtime: ", time.time()-time_start)
    return node_ids, node_embeddings

def EmbedPoincare(relations,epochs,dimension):
    model = PoincareModel(relations,size=dimension,workers=32)
    model.train(epochs)
    node_ids = model.index2entity
    node_embeddings = model.vectors
    return node_ids, node_embeddings