import random

def TraverseAndSelect(length, num_walks, hyperedges, vertexMemberships, p_select_method, p_select=0.15, p_select_initial=0.1):
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
                if p_select_method=="constant":
                    if initial:
                        proba = p_select_initial
                        initial=False
                    else:
                        proba = p_select
                if p_select_method=="inverse":
                    proba = float(1)/len(vertexMemberships[curr_vertex])
                if random.random()<proba:
                    adjacent_vertices = curr_hyperedge["members"]
                    curr_vertex = random.choice(adjacent_vertices)
                walk_hyperedge.append(str(curr_hyperedge_num))
                adjacent_hyperedges = vertexMemberships[curr_vertex]
                curr_hyperedge_num = random.choice(adjacent_hyperedges)
                curr_hyperedge = hyperedges[curr_hyperedge_num]
            walksTAS.append(walk_hyperedge)        
    return walksTAS

def SubsampleAndTraverse(length, num_walks, hyperedges, vertexMemberships, p_traverse_method, p_traverse=0.15, p_traverse_initial=0.1):
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
                if p_traverse_method=="constant":
                    if initial:
                        proba = p_traverse_initial
                        initial=False
                    else:
                        proba = p_traverse
                if p_traverse_method=="inverse":
                    proba = float(1)/len(curr_hyperedge["members"])
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