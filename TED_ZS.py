import numpy as np
from itertools import combinations_with_replacement
import os
import pandas as pd
import argparse



def obtain_nodes_all_attributes(dir, features_to_keep = [1,2,3,-1]):
    #add docstring

    node_attributes = []
    edges = []
    with open(dir, "r") as gml_file:
        lines = gml_file.readlines()
        for num_line, line in enumerate(lines):
            if line.startswith("node"):
                node_attributes.extend([
                    lines[num_line + 1].split()[1], #id 0
                    lines[num_line + 3].split()[1], #potential 1
                    lines[num_line + 4].split()[1], #area 2
                    lines[num_line + 5].split()[1], #volume 3
                    lines[num_line + 6].split()[1], #atom 4
                    lines[num_line + 7].split()[1], #x 5
                    lines[num_line + 8].split()[1], #y 6
                    lines[num_line + 9].split()[1], #z 7
                    lines[num_line + 10].split()[1],]) #izquierda 8 
            elif line.startswith("edge"):
                edges.extend([
                    lines[num_line + 1].split()[1],
                    lines[num_line + 2].split()[1]])

    node_attributes = np.array(node_attributes).reshape(-1, 9)
    edges = np.array(edges, dtype= np.int32).reshape(-1, 2)

    selected = node_attributes[:, features_to_keep]
    selected = np.asarray(selected, dtype=np.float64)
    return selected, edges



def build_children(edges, features):
    #add docstring

    n = len(features)
    children = [[] for _ in range(n)]

    for c, p in edges:
        children[p].append(c)

    #order children by attribute on last column
    for p in range(n):
        if len(children[p]) > 0:
            hijos = np.array(children[p])
            pesos = features[hijos][:, -1]
            idx = np.argsort(pesos)
            children[p] = list(hijos[idx])

    return children


def find_root(edges, n):
    #add docstring

    has_parent = np.zeros(n, dtype=bool)
    for c, _ in edges:
        has_parent[c] = True

    roots = np.where(~has_parent)[0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, but {len(roots)} were found")
    
    return int(roots[0])


def postorder_traversal(root, children):
    #add docstring, check original paper, other implementations might neer Euler tour

    order = []

    def dfs(u):
        for v in children[u]:
            dfs(v)
        order.append(u)

    dfs(root)
    return order


def compute_l_values(postorder, children):
    """
    l[i] = postorder index of leftmost leaf descendant of node postorder[i]
    """

    n = len(postorder)
    idx = {postorder[i]: i for i in range(n)}

    l = np.zeros(n, dtype=int)

    # For each node, follow first child until leaf
    for i in range(n):
        u = postorder[i]
        v = u
        while len(children[v]) > 0:
            v = children[v][0]
        l[i] = idx[v]

    return l


def compute_keyroots(l):
    #add docstring
    seen = set()
    keyroots = []

    for i in reversed(range(len(l))):
        if l[i] not in seen:
            keyroots.append(i)
            seen.add(l[i])

    return list(reversed(keyroots))




def zhang_shasha_ted(edges1, features1, edges2, features2):

    W_DISC = 1 #change if appropiate
    W_CONT = 1

    def normalize_joint(X, Y): #normalization is done among pairs, is it ok?

        Z = np.vstack([X, Y]).astype(float)
        mn = Z.min(axis=0)
        mx = Z.max(axis=0)
        den = mx - mn
        den[den == 0] = 1.0
        Z = (Z - mn) / den
        return Z[:len(X)], Z[len(X):]

    cont1, cont2 = normalize_joint(features1[:, 1:], features2[:, 1:])
    features1 = np.hstack([features1[:, [0]], cont1])
    features2 = np.hstack([features2[:, [0]], cont2])

    #BUild trees
    children1 = build_children(features=features1, edges=edges1)
    children2 = build_children(features=features2, edges=edges2)

    root1 = find_root(edges=edges1, n=len(features1))
    root2 = find_root(edges=edges2, n=len(features2))

    post1 = postorder_traversal(root=root1, children=children1)
    post2 = postorder_traversal(root=root2, children=children2)

    n1 = len(post1)
    n2 = len(post2)

    l1 = compute_l_values(postorder=post1, children=children1)
    l2 = compute_l_values(postorder=post2, children=children2)

    keyroots1 = compute_keyroots(l=l1)
    keyroots2 = compute_keyroots(l=l2)

    #cost functions that discriminate discrete and continuous variables
    def del_cost(i): 
        u = post1[i]
        return W_DISC * abs(features1[u][0]) + W_CONT * np.linalg.norm(features1[u][1:-1]) 
    def ins_cost(j):
        v = post2[j]
        return W_DISC * abs(features2[v][0]) + W_CONT * np.linalg.norm(features2[v][1:-1])

    def rep_cost(i, j):
        u = post1[i]
        v = post2[j]
        return (W_DISC* abs(features1[u][0] - features2[v][0]) + 
                W_CONT * np.linalg.norm(features1[u][1:-1] - features2[v][1:-1]))
    

    treedist = np.zeros((n1, n2))

    for i in keyroots1:
        for j in keyroots2:

            i0 = l1[i]
            j0 = l2[j]

            m = i - i0 + 2
            n = j - j0 + 2

            fd = np.zeros((m, n))

            #init forest distance
            for di in range(1, m):
                fd[di, 0] = fd[di - 1, 0] + del_cost(i0 + di - 1)
            for dj in range(1, n):
                fd[0, dj] = fd[0, dj - 1] + ins_cost(j0 + dj - 1)

            #DP
            for di in range(1, m):
                for dj in range(1, n):

                    ii = i0 + di - 1
                    jj = j0 + dj - 1

                    if l1[ii] == i0 and l2[jj] == j0:
                        fd[di, dj] = min(
                            fd[di - 1, dj] + del_cost(ii),
                            fd[di, dj - 1] + ins_cost(jj),
                            fd[di - 1, dj - 1] + rep_cost(ii, jj)
                        )
                        treedist[ii, jj] = fd[di, dj]
                    else:
                        fd[di, dj] = min(
                            fd[di - 1, dj] + del_cost(ii),
                            fd[di, dj - 1] + ins_cost(jj),
                            fd[l1[ii] - i0, l2[jj] - j0] + treedist[ii, jj]
                        )

    return treedist[n1 - 1, n2 - 1]


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Calculates TED using Zhang-Shasha's algorithm combining both discrete" \
    "and continuous variables")

    parser.add_argument("folder", type=str, help="Directory in which the gml files are contained")
    parser.add_argument("csv_out", type=str, default=None, help="Directory in which the csv file should be outputed")
    args = parser.parse_args()

    dir = args.folder
    dir_list = list(os.listdir(dir)) 
    dir_list.sort()
    n = len(dir_list)
    dm = np.full((n, n), np.nan)

    labels = [archivo[:-4] for archivo in dir_list]

    for i, j in combinations_with_replacement(range(n), 2):
        file_i = os.path.join(dir, dir_list[i])
        file_j = os.path.join(dir, dir_list[j])


        features_i, edges_i = obtain_nodes_all_attributes(file_i)
        features_j, edges_j = obtain_nodes_all_attributes(file_j)

        ted = zhang_shasha_ted(edges_i, features_i, edges_j, features_j)

        print(f"ted({labels[i]}, {labels[j]}) = {ted}")
        dm[i, j] = round(ted, 5)
        dm[j, i] = round(ted, 5)
    
    df = pd.DataFrame(dm, index=labels, columns=labels)
    csv_path = args.csv_out
    df.to_csv(csv_path)
    print(f"Distance matrix saved to {csv_path}")



