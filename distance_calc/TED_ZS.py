import numpy as np
from itertools import combinations_with_replacement
import os
import pandas as pd
import argparse


FEATURE_SCHEMA = {
    "id":        {"index": 0, "type": "meta"}, 
    "potential": {"index": 1, "type": "discrete"},
    "area":      {"index": 2, "type": "continuous"},
    "volume":    {"index": 3, "type": "continuous"},
    "atom":      {"index": 4, "type": "discrete"}, #NOT HANDLED
    "x":         {"index": 5, "type": "continuous"},
    "y":         {"index": 6, "type": "continuous"},
    "z":         {"index": 7, "type": "continuous"},
    "izquierda": {"index": 8, "type": "meta"}}


def obtain_nodes_all_attributes(dir, features_to_keep=("potential","area","volume","izquierda")):
    """Stores for all the nodes the values found in features to keep, 
    the value -1 should always be store to handle the canonical arrangement
    the algorithm utilizes"""

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
    edges = np.array(edges, dtype=np.int32).reshape(-1, 2)

    indices = [FEATURE_SCHEMA[f]["index"] for f in features_to_keep]
    types   = [FEATURE_SCHEMA[f]["type"]  for f in features_to_keep]

    selected = node_attributes[:, indices].astype(np.float64)

    return selected, edges, types


def build_children(edges, features):

    """Generates a list indicating the ordered children of each node"""

    n = len(features)
    children = [[] for _ in range(n)]

    for c, p in edges:
        children[p].append(c)

    #order children by attribute on last column
    for p in range(n):
        if len(children[p]) > 0:
            children_np = np.array(children[p])
            weights = features[children_np][:, -1]
            idx = np.argsort(weights)
            children[p] = list(children_np[idx])

    return children


def find_root(edges, n):
    """Finds the root of the tree"""

    has_parent = np.zeros(n, dtype=bool)
    for c, _ in edges:
        has_parent[c] = True

    roots = np.where(~has_parent)[0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, but {len(roots)} were found")
    
    return int(roots[0])


def postorder_traversal(root, children):
    """Postorder traversal required for the greedy implementation of TED, based on the work
    of Kaizhong Zhang & Dennis Shasha https://doi.org/10.1137/0218082 """

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
    """Compute keyroots of the graphs"""

    seen = set()
    keyroots = []

    for i in reversed(range(len(l))):
        if l[i] not in seen:
            keyroots.append(i)
            seen.add(l[i])

    return list(reversed(keyroots))




def zhang_shasha_ted(edges1, features1, types1, edges2, features2, types2):
    """Zhang & Shasha tree edit distance calculation weighted by the type of feature"""

    W_DISC = 1 #change if appropiate
    W_CONT = 1

    def normalize_joint(X, Y): #normalization is done among pairs

        Z = np.vstack([X, Y]).astype(float)
        mn = Z.min(axis=0)
        mx = Z.max(axis=0)
        den = mx - mn
        den[den == 0] = 1.0
        Z = (Z - mn) / den
        return Z[:len(X)], Z[len(X):]
    
    disc_idx = [i for i,t in enumerate(types1) if t == "discrete"]
    cont_idx = [i for i,t in enumerate(types1) if t == "continuous"]
    meta_idx = [i for i,t in enumerate(types1) if t == "meta"]

    f1_disc = features1[:, disc_idx] if disc_idx else np.empty((len(features1),0))
    f2_disc = features2[:, disc_idx] if disc_idx else np.empty((len(features2),0))

    f1_cont = features1[:, cont_idx] if cont_idx else np.empty((len(features1),0))
    f2_cont = features2[:, cont_idx] if cont_idx else np.empty((len(features2),0))

    f1_meta = features1[:, meta_idx] if meta_idx else np.empty((len(features1),0))
    f2_meta = features2[:, meta_idx] if meta_idx else np.empty((len(features2),0))

    if f1_cont.shape[1] > 0:
        f1_cont, f2_cont = normalize_joint(f1_cont, f2_cont)

    features1 = np.hstack([f1_disc, f1_cont, f1_meta])
    features2 = np.hstack([f2_disc, f2_cont, f2_meta])

    n_disc = f1_disc.shape[1]
    n_cont = f1_cont.shape[1]

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
        disc = np.sum(np.abs(features1[u][:n_disc])) if n_disc else 0
        cont = np.linalg.norm(features1[u][n_disc:n_disc+n_cont]) if n_cont else 0
        return W_DISC * disc + W_CONT * cont

    def ins_cost(j):
        v = post2[j]
        disc = np.sum(np.abs(features2[v][:n_disc])) if n_disc else 0
        cont = np.linalg.norm(features2[v][n_disc:n_disc+n_cont]) if n_cont else 0
        return W_DISC * disc + W_CONT * cont

    def rep_cost(i, j):
        u = post1[i]
        v = post2[j]

        disc = np.sum(np.abs(features1[u][:n_disc] -
                     features2[v][:n_disc])) if n_disc else 0

        cont = np.linalg.norm(features1[u][n_disc:n_disc+n_cont] -
                              features2[v][n_disc:n_disc+n_cont]) if n_cont else 0

        return W_DISC * disc + W_CONT * cont

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

    labels = [file[:-4] for file in dir_list]

    for i, j in combinations_with_replacement(range(n), 2):
        file_i = os.path.join(dir, dir_list[i])
        file_j = os.path.join(dir, dir_list[j])

        features_extract = ("potential","volume","izquierda") #CHANGE ACCORDINGLY
        features_i, edges_i, types_i = obtain_nodes_all_attributes(file_i, features_extract)
        features_j, edges_j, types_j = obtain_nodes_all_attributes(file_j, features_extract)

        ted = zhang_shasha_ted(edges1=edges_i, features1=features_i, types1=types_i,
                               edges2=edges_j, features2=features_j, types2=types_j)

        print(f"ted({labels[i]}, {labels[j]}) = {ted}")
        dm[i, j] = round(ted, 5)
        dm[j, i] = round(ted, 5)
    
    df = pd.DataFrame(dm, index=labels, columns=labels)
    csv_path = args.csv_out
    df.to_csv(csv_path)
    print(f"Distance matrix saved to {csv_path}")



