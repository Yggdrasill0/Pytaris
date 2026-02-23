import networkx as nx
import graphviz as gv
import os
import re


color_dir = {"H": "#FFFFFF", "C": "#000000", "N": "#00008B",
             "O": "#8B0000", "F" : "#32CD32", "Cl": "#006400"}


def graph_viz_visualization(dir):

    dot = gv.Digraph(format="png")
    basename = re.sub( r"\.(.*)", "", os.path.basename(dir))

    G = nx.read_gml(dir)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    for node, attr in G.nodes(data= True):
        potential =float(attr["potentialValue"])
        atom_type = re.sub( r"\d", "", attr["atomtype"])
        is_leaf = G.in_degree(node)
        atom_label = atom_type if (is_leaf == 0) else ""
        label = f"{atom_label}\nPot={potential}"

        dot.node(str(node), label=label, style="filled", fillcolor = color_dir.get(atom_label, "#808080"),
                shape = "ellipse" if (atom_label == "" ) else "box",
                fontcolor = "#000000" if (atom_label != "C") else "#FFFFFF")

    for u, v in G.edges():
        dot.edge(str(v), str(u))
   

    dot.attr(rankdir="TB")
    dot.render(f"{basename}", view=True)

    






