from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec
import generate
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


max_h = generate.max_hyperedge_size

num_positive = generate.num_positive_hyperedges
num_negative = generate.num_negative_hyperedges
nx_graph = generate.getGraph()

def graphNodeEmbed(dimension):    
    rw = BiasedRandomWalk(StellarGraph(nx_graph))

    walks = rw.run(nodes=list(nx_graph.nodes()),length=5,n=25,p=0.5,q=2.0)
    print("Number of graph random walks: {}".format(len(walks)))

    model = Word2Vec(walks, size=dimension, window=5, min_count=0, sg=1, workers=16, iter=1)

    node_ids = model.wv.index2word  # list of node IDs
    
    node_embeddings_graph = model.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [ nx_graph.node[node_id]['type'] for node_id in node_ids]
    # print(node_targets)
    print("Graph embeddings generated.")
    
    ####### VISUALIZATION #######

    transform = TSNE

    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings_graph)

    alpha = 0.7
    label_map = { l: i for i, l in enumerate(np.unique(node_targets))}

    node_colours = [ label_map[target] for target in node_targets]

    plt.figure(figsize=(7,7))
    plt.axes().set(aspect="equal")
    plt.scatter(node_embeddings_2d[:,0], 
                node_embeddings_2d[:,1], 
                c=node_colours, cmap="jet", alpha=alpha)
    plt.title('{} visualization of vertex embeddings in the graph.'.format(transform.__name__))
    plt.show()
    
    ####### VISUALIZATION #######

    return node_embeddings_graph


# ### Hyperedge embedding ###

def generateHypergraphWalks(length, num_walks, p_traverse):
        hyperedges = generate.getHyperedges()
        walks_node = []
        walk_labels_node = []
        walks_hyperedge = []
        walk_labels_hyperedge = []
        label = 0
        for hyperedge in hyperedges:
            walk_hyperedge = []
            walk_node = []
            curr_node = random.choice(hyperedge)
            for _ in range(num_walks):
                for i in range(length):
                    curr_hyperedge = hyperedge
                    if random.random()<p_traverse:
                        adjacent_hyperedges = [h_e for h_e in hyperedges if (curr_node in h_e)]
                        curr_hyperedge = random.choice(adjacent_hyperedges)
                    curr_node = random.choice(curr_hyperedge)
                    walk_hyperedge.append(str(label))
                    walk_node.append(str(curr_node))
                walks_hyperedge.append(walk_hyperedge)
                walks_node.append(walk_node)
                if (label<num_negative):
                    walk_labels_node.append("negative")
                else:
                    walk_labels_node.append("positive")
            if (label<num_negative):
                walk_labels_hyperedge.append("negative")
            else:
                walk_labels_hyperedge.append("positive")
            label+=1
        return walks_hyperedge, walk_labels_hyperedge, walks_node, walk_labels_node

def hyperedgeEmbed(hg_walks, walk_labels, dimension):

    model = Word2Vec(hg_walks, size=dimension, window=5, min_count=0, sg=1, workers=16, iter=1)

    node_ids = model.wv.index2word  # list of node IDs
    node_embeddings_hyperedges = model.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = walk_labels
    print("Hypergraph embeddings generated.")
    ####### VISUALIZATION #######
    transform = TSNE

    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings_hyperedges)

    alpha = 0.7
    label_map = { l: i for i, l in enumerate(np.unique(node_targets))}

    node_colours = [ label_map[target] for target in node_targets]

    plt.figure(figsize=(7,7))
    plt.axes().set(aspect="equal")
    plt.scatter(node_embeddings_2d[:,0], 
                node_embeddings_2d[:,1], 
                c=node_colours, cmap="jet", alpha=alpha)
    plt.title('{} visualization of hyperedge embeddings in the hypergraph.'.format(transform.__name__))
    plt.show()
    ####### VISUALIZATION #######
    return node_embeddings_hyperedges
    
def hypergraphNodeEmbed(hg_walks, walk_labels, dimension):
    
    model = Word2Vec(hg_walks, size=dimension, window=5, min_count=0, sg=1, workers=16, iter=1)

    node_ids = model.wv.index2word  # list of node IDs
    node_embeddings_hypergraph = model.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [nx_graph.node[node_id]['type'] for node_id in node_ids]
    print("Hypergraph embeddings generated.")
    
    ####### VISUALIZATION #######
    transform = PCA

    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings_hypergraph)

    alpha = 0.7
    label_map = { l: i for i, l in enumerate(np.unique(node_targets))}

    node_colours = [ label_map[target] for target in node_targets]

    plt.figure(figsize=(7,7))
    plt.axes().set(aspect="equal")
    plt.scatter(node_embeddings_2d[:,0], 
                node_embeddings_2d[:,1], 
                c=node_colours, cmap="jet", alpha=alpha)
    plt.title('{} visualization of vertex embeddings in the hypergraph.'.format(transform.__name__))
    plt.show()
    
    transform = TSNE

    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings_hypergraph)

    alpha = 0.7
    label_map = { l: i for i, l in enumerate(np.unique(node_targets))}

    node_colours = [ label_map[target] for target in node_targets]

    plt.figure(figsize=(7,7))
    plt.axes().set(aspect="equal")
    plt.scatter(node_embeddings_2d[:,0], 
                node_embeddings_2d[:,1], 
                c=node_colours, cmap="jet", alpha=alpha)
    plt.title('{} visualization of vertex embeddings in the hypergraph.'.format(transform.__name__))
    plt.show()
    
#     trans = transform(n_components=3)
#     node_embeddings_3d = trans.fit_transform(node_embeddings_hypergraph)
#     fig = plt.figure()
#     d3 = fig.add_subplot(111, projection='3d')
#     d3.scatter(*zip(*node_embeddings_3d),c=node_colours)
#     plt.show()
    ####### VISUALIZATION #######
    return node_embeddings_hypergraph
    
embedding_dimension = 128
length = 20
num_walks = 20
p_traverse = 0.1
walks_hyperedge, walk_labels_hyperedge, walks_node, walk_labels_node = generateHypergraphWalks(length, num_walks, p_traverse)
print("Number of hypergraph random walks: {}".format(len(walks_hyperedge)))
print("Generating graph embeddings...")
node_embeddings_graph = graphNodeEmbed(embedding_dimension)
print("Generating hyperedge embeddings...")
hyperedge_embeddings_hypergraph = hyperedgeEmbed(walks_hyperedge, walk_labels_hyperedge, embedding_dimension)
print("Generating hypergraph node embeddings...")
node_embeddings_hypergraph = hypergraphNodeEmbed(walks_node, walk_labels_node, embedding_dimension)

def getGraphNodeEmbeddingHyperedges():
    embeddinglists = []
    print(len(node_embeddings_graph))
    hyperedges = generate.getHyperedges()
    for hyperedge in hyperedges:
        embeddinglist = []
        for node in hyperedge:
            embeddinglist.append(node_embeddings_graph[node].tolist())
        embeddinglists.append(embeddinglist)
    return embeddinglists

def getHypergraphNodeEmbeddingHyperedges():
    embeddinglists = []
    hyperedges = generate.getHyperedges()
    for hyperedge in hyperedges:
        embeddinglist = []
        for node in hyperedge:
            embeddinglist.append(node_embeddings_hypergraph[node].tolist())
        embeddinglists.append(embeddinglist)
    return embeddinglists

def getHyperedgeEmbeddingHyperedges():
    embeddinglists = []
    hyperedges = generate.getHyperedges()
    for hyperedge in hyperedges:
        embeddinglist = []
        for node in hyperedge:
            embeddinglist.append(hyperedge_embeddings_hypergraph[node].tolist())
        embeddinglists.append(embeddinglist)
    return embeddinglists

