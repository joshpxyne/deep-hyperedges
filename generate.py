import numpy as np
import random
import networkx as nx
import requests

num_assets = 150
num_users = 150
intersection_prob = 0.03
asset_user_split = 0.5
connectivity = 0.05

min_hyperedge_size = 2
max_hyperedge_size = 15

num_positive_hyperedges = 400
num_negative_hyperedges = 600

edgeset = set([])
userlist = []
assetlist = []

assets = range(num_assets)
users = range(num_assets,num_users + num_assets)

multiuser_groups = []
intersection_groups = []

print("Generating graph...")

for asset in assets:
    group = [asset]
    for num in range(random.randint(0,num_assets)): # first few dbs have higher chance
        if random.random()<intersection_prob:
            group.append(num)
    if len(group)>1:
        intersection_groups.append(group)

pairs = []
triples = []
quads = []

def groups(grouplist, n):
    ret = []
    while len(grouplist)>n:
        group = []
        for _ in range(n):
            group.append(grouplist.pop())
        ret.append(group)
    return ret

for user in users:
    if random.random()<0.07:
        pairs.append(user)
    elif random.random()<0.05:
        triples.append(user)
    elif random.random()<0.05:
        quads.append(user)

multiuser_groups = groups(pairs,2)+groups(triples,3)+groups(quads,4)

def makeNegativeHyperedges(num):
    hyperedges = []
    for _ in range(num):
        probabilities = [asset_user_split*1/float(num_users)]*num_users + [(1-asset_user_split)*1/float(num_assets)]*num_assets
        size = random.randint(min_hyperedge_size,max_hyperedge_size)
        hyperedge = np.random.choice(range(num_users+num_assets),size,replace=False,p=probabilities).tolist()
        if (min(hyperedge)>num_assets):
            hyperedge.append(np.random.choice(range(num_assets),1)[0])
        if (max(hyperedge)<num_assets):
            hyperedge.append(np.random.choice(range(num_assets,num_users + num_assets),1)[0])
        hyperedges.append(hyperedge)
    return hyperedges

def makePositiveHyperedges(num):
    hyperedges = []
    for _ in range(num):
        hyperedge = np.random.choice(multiuser_groups)+np.random.choice(intersection_groups)
        hyperedges.append(hyperedge)
    return hyperedges


hyperedges = makeNegativeHyperedges(num_negative_hyperedges)+makePositiveHyperedges(num_positive_hyperedges)

G = nx.Graph()

for node in range(num_users+num_assets):
    reqString = ""
    if node < num_assets:
        assetlist.append(asset)
        G.add_node(str(node),type="asset")
        reqString = "http://localhost:9000/createAsset?assetID=dataset_"+str(node)
    else:
        userlist.append(node)
        G.add_node(str(node),type="user")
        reqString = "http://localhost:9000/createClient?clientID=user_"+str(node)
    r = requests.get(reqString, data="none")
    print(r.status_code)

# add edges from hyperedges

for hyperedge in hyperedges:
    userset = [i for i in hyperedge if i >= num_assets]
    assetset = [i for i in hyperedge if i < num_assets]
    for user in userset:
        for asset in assetset:
            edgeset.add((user,asset))

for user in userlist:
    for asset in assetlist:
        if random.random()<connectivity:
            edgeset.add((user,asset))

for edge in edgeset:
    edge0 = str(edge[0])
    edge1 = str(edge[1])
    G.add_edge(edge0,edge1)
    queryString = "http://localhost:9000/createEdge"
    queryString+="?fromID=user_"+edge0
    queryString+="&toID=dataset_"+edge1
    queryString+="&fromType=client"
    queryString+="&toType=asset"
    queryString+="&label=requests"
    r = requests.get(queryString, data="")
    print(r.status_code)

print("Graph generated.")
print("Graph statistics: {} nodes, {} edges".format(G.number_of_nodes(), G.number_of_edges()))
nx.write_adjlist(G,"top.adjlist")

def getHyperedges():
    return hyperedges

def getGraph():
    return G