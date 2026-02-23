from utils import collapse_tree_pytaris, marching_cubes
from visualization import from_gml_2_graph_viz
import sys
sys.setrecursionlimit(int(1E8))
import os
import argparse
import re
import numpy as np



def main():
    parser = argparse.ArgumentParser(description="Process gaussian .cube for the generation of .gml files")
    parser.add_argument("path_cube", type=str, help="Route to .cube file")


    parser.add_argument("--neg_min", type=float, default=None, help="Most negative potential (neg_start)")
    parser.add_argument("--neg_max", type=float, default=-0.01, help="Least nega potential (neg_end)")
    parser.add_argument("--neg_step", type=float, default=0.002, help="Potential step for the negative tree")

    parser.add_argument("--pos_min", type=float, default=0.1, help="Least positive potential  (pos_end)")
    parser.add_argument("--pos_max", type=float, default=1.5, help="Most positive potential value (pos_begin)")
    parser.add_argument("--pos_step", type=float, default=-0.05, help="Potential step for the negative tree")
    parser.add_argument("--visualize", type=bool, default= False, help ="Visualize and store all the generated graphs")

    args = parser.parse_args()

    dir = args.path_cube
    basename = re.sub( r"\.(.*)", "", os.path.basename(dir))

    ### Negative cube
    cache = {}
    Neg_cube = True
    atom_labels, atom_coords = marching_cubes.extract_coordinates_cube_ordered_atoms(dir)


    values_pem, delta_x, delta_y, delta_z, origin, min_value = marching_cubes.obtain_coordinates(dir, Neg_cube)
    values_isopotential = marching_cubes.obtain_values_neg(P_min=args.neg_min, P_max=args.neg_max, step=args.neg_step, min_value=min_value)
    conec, dicc_pytaris = marching_cubes.Pytaris(
    values_isopotential,
    values_pem,
    delta_x, delta_y, delta_z,
    origin,
    negative=True,
    atom_coords=atom_coords,
    atom_labels=atom_labels)

    keys, nodes_data, atom_types = marching_cubes.preprocess_nodes(dicc_pytaris, reverse=False)


    nodes_data_left = marching_cubes.left_tag_nodes(nodes_data, keys)
    edges = marching_cubes.obtain_global_pre_edges(dicc_pytaris, conec)    
    edges = marching_cubes.obtain_global_edges(edges, nodes_data)
    file_name = basename + "_neg.gml"
    marching_cubes.create_file(file_name=file_name, nodes_data=nodes_data_left, atom_types=atom_types, edge_list=edges)

    ### Positive cube
    cache = {}
    Neg_cube = False

    values_pem, delta_x, delta_y, delta_z, origin = marching_cubes.obtain_coordinates(dir, Neg_cube)
    values_isopotential = marching_cubes.obtain_values_pos(P_max=args.pos_max, P_min=args.pos_min, step=args.pos_step)
    conec, dicc_pytaris = marching_cubes.Pytaris(
    values_isopotential,
    values_pem,
    delta_x, delta_y, delta_z,
    origin,
    negative=False,
    atom_coords=atom_coords,
    atom_labels=atom_labels)

    keys, nodes_data, atom_types = marching_cubes.preprocess_nodes(dicc_pytaris, reverse=True)
    nodes_data_left = marching_cubes.left_tag_nodes(nodes_data, keys)
    edges = marching_cubes.obtain_global_pre_edges(dicc_pytaris, conec)    
    edges = marching_cubes.obtain_global_edges(edges, nodes_data)
    file_name = basename + "_pos.gml"
    marching_cubes.create_file(file_name=file_name, nodes_data=nodes_data_left, atom_types=atom_types, edge_list=edges)


    with open(file_name ,"r") as gml_file:
        pairs = []
        lines = gml_file.readlines()
        area_values = []

        for i in range(len(lines)):
            if lines[i].startswith("source"):
                source = lines[i].split()[-1]
                target = lines[i+1].split()[-1]
                pairs.append([source, target])

            elif lines[i].startswith("areaValue"):
                area_values.append(float(lines[i].split()[-1]))


        adj_array = np.array(pairs, dtype="int64")
        
        output_file = os.path.join(basename + "_pos_col.gml")
        
        if adj_array.size == 0:
            with open(output_file, "w") as out:
                out.write("graph [\n")
                out.write("directed 1\n")
                out.write("node [\n")
                out.write("id 0\n")
                out.write("label \"0\"\n")
                out.write("potentialValue 0.0\n")
                out.write("areaValue 0.0\n")
                out.write("volumeValue 0.0\n")
                out.write("atomtype  'ROOT'\n")
                out.write("xValue 0.0\n")
                out.write("yValue 0.0\n")
                out.write("zValue 0.0\n")
                out.write("izquierda 0\n")
                out.write("]\n")
                out.write("]\n")
                out.write("tree 1\n")
                out.write("]\n")

        root = collapse_tree_pytaris.root_node(adj_array, filename=basename)
        critical_nodes_gml = collapse_tree_pytaris.critical_nodes(adj_array, root)
        collapsed_edges = collapse_tree_pytaris.collapse_tree_adj(adj_array, critical_nodes_gml)

        
        collapse_tree_pytaris.write_collapsed_tree(file_name, output_file, critical_nodes_gml, collapsed_edges)
    
    if args.visualize:
        from_gml_2_graph_viz.graph_viz_visualization(basename + "_pos_col.gml")
        from_gml_2_graph_viz.graph_viz_visualization(basename + "_neg.gml")


if __name__ == "__main__":
    main()


#...............................................................
#...............................................................
#...............................................................
#...............................................................
#.......................................:.......................
#...........................:.@.........#@......................
#............................#@@.........:......................
#......................*................-.......................
#........................@.....@...@@.=.........................
#...........................-@.@.....+..........................
#..........................%..@@@.@+@...........................
#.............................@@.@.@.....:..........%...........
#................#..........%..@.-=.........=....@..............
#............-.................@.@@.:......@..=.................
#...................@..........@@...*@......@...................
#...............=.++@.%@@@:@...@*@:......@.@=...................
#............................@.@@.......@.......................
#.............................@@*...+@.@..#@.*..................
#.............................@@@....%@.........................
#..............................@@+.@@...........................
#.............................@@@@@.............................
#.............................@@@...............................
#.............................-@@...............................
#.............................@@@...............................
#.............................@@@...............................
#.............................@@@@..............................
#...........................@@@@@@@.............................
#...............................................................
#...............................................................

