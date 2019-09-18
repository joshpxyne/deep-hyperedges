import random

def TraverseAndSelect(length, num_walks, hyperedges, vertexMemberships, alpha=1., beta=0):
    walksTAS = []
    for hyperedge_index in hyperedges:
        hyperedge = hyperedges[hyperedge_index]
        walk_hyperedge = []
        for _ in range(num_walks):
            curr_vertex = random.choice(hyperedge["members"])
            initial=True
            curr_hyperedge_num = hyperedge_index
            curr_hyperedge = hyperedge
            for i in range(length): 
                proba = (float(alpha)/len(vertexMemberships[curr_vertex])) + beta
                if random.random()<proba:
                    adjacent_vertices = curr_hyperedge["members"]
                    curr_vertex = random.choice(adjacent_vertices)
                walk_hyperedge.append(str(curr_hyperedge_num))
                adjacent_hyperedges = vertexMemberships[curr_vertex]
                curr_hyperedge_num = random.choice(adjacent_hyperedges)
                curr_hyperedge = hyperedges[curr_hyperedge_num]
            walksTAS.append(walk_hyperedge)        
    return walksTAS

def SubsampleAndTraverse(length, num_walks, hyperedges, vertexMemberships, alpha=1., beta=0):
    walksSAT = []
    for hyperedge_index in hyperedges:
        hyperedge = hyperedges[hyperedge_index]
        walk_vertex = []
        curr_vertex = random.choice(hyperedge["members"])
        for _ in range(num_walks):
            initial=True
            hyperedge_num = hyperedge_index
            curr_hyperedge = hyperedge
            for i in range(length):
                proba = (float(alpha)/len(curr_hyperedge["members"])) + beta
                if random.random()<proba:
                    adjacent_hyperedges = vertexMemberships[curr_vertex]
                    hyperedge_num = random.choice(adjacent_hyperedges)
                    curr_hyperedge = hyperedges[hyperedge_num]
                walk_vertex.append(str(curr_vertex))
                curr_vertex = random.choice(curr_hyperedge["members"])
            walksSAT.append(walk_vertex)        
    return walksSAT

if __name__ == "__main__":
    pass