import numpy as np
import os


def root_node(adj_array, filename="Unknown file"):
    """Returns the root node of a tree, generates an error if graph is not a tree"""
    sons = set(adj_array[:, 0])
    fathers = set(adj_array[:, -1])

    assert len(fathers - sons) == 1, f"There is an error in the processing of file: {filename}"

    return (fathers - sons).pop()


def critical_nodes(adj_array, root):
  """Returns the nodes having 0 or more than 1 children"""

  unique_nodes = np.unique(adj_array)
  critical_nodes = []

  for node in unique_nodes:
    num_children = np.sum(adj_array[:, 1] == node)

    if num_children != 1:
      critical_nodes.append(node)

  critical_nodes = np.sort(np.array(critical_nodes))
  if critical_nodes[-1] == root:
    return critical_nodes
  else:
    return np.append(critical_nodes, root)


def collapse_tree_adj(adj_array, critical_nodes):
  """Simplifies the graph obtaining a subgraph of the original one posessing only critical nodes"""

  collapsed_adj_array = []

  for critical_node in critical_nodes[:-1]:
    current_node = critical_node

    while True:
      father = adj_array[adj_array[:, 0] == current_node][0][1]
      if father in critical_nodes:
        collapsed_adj_array.append([critical_node , father])
        break
      else:
        current_node = father

  collapsed_adj_array = np.array(collapsed_adj_array)
  return collapsed_adj_array


def write_collapsed_tree(input_file, output_file, critical_nodes, collapsed_edges):

  with open(input_file, 'r') as infile:
    lines = infile.readlines()

    node_blocks = []
    current_block = []
    recording = False
    current_id = None

    for line in lines:
      stripped = line.strip()

      if stripped.startswith("node ["):
        recording = True
        current_block = [line]
        current_id = None

      elif recording:
        current_block.append(line)

        if stripped.startswith("id"):
          current_id = int(stripped.split()[1])

        if stripped == "]":
          if current_id in critical_nodes:
            node_blocks.append("".join(current_block))
          recording = False
          current_block = []

    with open(output_file, 'w') as outfile:
      outfile.write("graph [ \n")
      outfile.write("directed 1 \n")

      for block in node_blocks:
        outfile.write(block + "\n")

      for edge in collapsed_edges:
        outfile.write("edge [ \n")
        outfile.write(f"source {edge[0]} \n")
        outfile.write(f"target {edge[1]} \n")
        outfile.write("\t ] \n")
      outfile.write("tree 1 \n")
      outfile.write("] \n")


INPUT_PATH = "~/Desktop/tmp_graphs/new_files/"
INPUT_PATH = os.path.expanduser(INPUT_PATH)
OUTPUT_PATH = "~/Desktop/tmp_graphs/tree_mols/"
OUTPUT_PATH = os.path.expanduser(OUTPUT_PATH)

os.makedirs(OUTPUT_PATH, exist_ok=True)

  



