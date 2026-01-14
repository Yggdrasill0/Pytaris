import os
import numpy as np
import re
from collections import defaultdict
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  


periodic_table_translator = {1:"H", 6: "C", 12: "Mg", 13: "Al", 8: "O", 7: "N", 29: "Cu", 27: "Co"}

CPK_COLORS = {"H":  "#FFFFFF", "C":  "#909090", "N":  "#3050F8", "O":  "#FF0D0D",
    "F":  "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I":  "#940094", "S":  "#FFFF30", "P":  "#FF8000",
    "Cu": "#C88033", "Zn": "#7D80B0", "Co" : "#03045E"}

def extract_coordinates_cube_ordered_atoms(cube_file: os.path):
    with open(cube_file) as cube:
        lines = cube.readlines()

    natoms = abs(int(lines[2].split()[0]))  
    start = 6                            
    end = start + natoms

    coords = []
    atoms = []
    for line in lines[start:end]:
        atom,_ , x, y, z = map(float, line.split())
        coords.append([x, y, z])
        atoms.append(periodic_table_translator[atom])

    numbering_dic = {}
    atoms_with_numbers = []
    for atom in atoms:
        if atom not in numbering_dic.keys():
            numbering_dic[atom] = 0
        else:
            numbering_dic[atom] += 1
        atoms_with_numbers.append(f"{atom}{numbering_dic[atom] + 1}")

    return atoms_with_numbers, np.array(coords)


def extract_coordinates_desired_potential(gml_file: str, potential: float):
    with open(gml_file, "r") as f:
        text = f.read()

    
    blocks = re.findall(r"node\s*\[\s*(.*?)\s*\]", text, flags=re.DOTALL)

    areas = []
    coordinates = []
    atom_types = []

    for block in blocks:
        pot = re.search(r"potentialValue\s+([-\d.eE]+)", block)
        if not pot:
            continue

        if not np.isclose(float(pot.group(1)), potential):
            continue

        volume = re.search(r"volumeValue\s+([-\d.eE]+)", block)
        x = re.search(r"xValue\s+([-\d.eE]+)", block)
        y = re.search(r"yValue\s+([-\d.eE]+)", block)
        z = re.search(r"zValue\s+([-\d.eE]+)", block)
        atom_type = re.search(r'atomtype\s+"([^"]+)"', block)

        if volume and x and y and z and atom_type:
            areas.append(round(float(volume.group(1)), 5))
            coordinates.append((
                float(x.group(1)),
                float(y.group(1)),
                float(z.group(1))
            ))
            atom_types.append(atom_type.group(1))

    if not areas:
        raise ValueError(f"The value {potential} is not a valid potential")

    return areas, np.array(coordinates), atom_types



def obtain_total_value_atom(values, atom_types):
    totals = defaultdict(float)
    for value, atom in zip(values, atom_types):
        totals[atom] += value
    return dict(totals)




"""
def plot_in_3d(atom_labels, atom_coors, node_coors, label_offset=0.2):
    atom_coors = np.asarray(atom_coors)
    node_coors = np.asarray(node_coors)
    atom_labels = np.asarray(atom_labels)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

 
    for label, (x, y, z) in zip(atom_labels, atom_coors):
  
        element = ''.join(filter(str.isalpha, label))
        color = CPK_COLORS.get(element, "#808080")

        ax.scatter(x, y, z, color=color, s=120, edgecolors="k")

        ax.text(x + label_offset, y + label_offset, z + label_offset,
            label, fontsize=9)

    ax.scatter(node_coors[:, 0], node_coors[:, 1], node_coors[:, 2], color="green", s=50, label="Nodes")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    plt.tight_layout()
    plt.show()"""



DIR_cube = r"/home/ricardo/Downloads/nicolas_cubes/su_1.cube"
DIR_gml = r"/home/ricardo/Downloads/nicolas_cubes/su_1_neg_Cuts.gml"

atoms, coors_cube = extract_coordinates_cube_ordered_atoms(DIR_cube)

volumes, coors_gml, atom_types_gml = extract_coordinates_desired_potential(DIR_gml, -0.004) #SE CAMBIO ESTO


print(obtain_total_value_atom(volumes, atom_types=atom_types_gml))


             

