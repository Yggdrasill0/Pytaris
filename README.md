# Pytaris: Molecular Electrostatic Potential Isosurface Tree 

A Python package for generating tree representations from Molecular Electrostatic Potential (MESP) isosurfaces computed from Gaussian cube files. The package implements marching cubes algorithm to extract isosurfaces and constructs hierarchical trees based on isopotential connectivity.

# Overview

Pytaris processes Gaussian cube files containing MESP data to generate directed acyclic graphs (trees) representing the topological evolution of isosurfaces across different potential values. The algorithm:

1.    Reads MESP data from Gaussian cube files

2.    Extracts isosurfaces for specified potential thresholds using marching cubes

3.    Identifies connected components (islands) at each potential level

4.    Tracks connectivity between components across potential levels

5.   Constructs a tree structure representing the hierarchical relationship of isosurfaces

6.    Outputs the tree in GML format for visualization and further analysis

# Features

*    Parallel processing of multiple isopotential values

*    Support for both negative and positive potential ranges

*    Automatic detection and tracking of connected components

*    Calculation of surface properties (area, volume, centroid)

*    Atom-type assignment to surface components based on nearest atom

*    GML file generation with complete node and edge attributes

*    Tree collapsing functionality to identify critical nodes

# Usage
## Command Line

```python

python main.py <path_to_cube_file> [options]
```

# Arguments

| Argument	 | Description |	Default |
| --- | --- | ---|
|path_cube |	Path to Gaussian .cube file (required) |	-
|--neg_min |Most negative potential (start value) |	None (auto-calculated)
|--neg_max |	Least negative potential (end value) |	-0.01
|--neg_step|	Step size for negative potential sweep |	0.002
|--pos_min |	Least positive potential (end value) |	0.1
|--pos_max | 	Most positive potential (start value) |	1.5
|--pos_step|	Step size for positive potential sweep |	-0.05
|--visualize |	Visualize generated graphs |	False

# Output Format

The generated GML files contain:
Node Attributes

*   id: Unique node identifier
*   label: Node label (same as id)
*   potentialValue: Isopotential value
* areaValue: Surface area of the component
*    volumeValue: Enclosed volume
* atomtype: Nearest atom type (e.g., "H1", "O2") or "ROOT"
* xValue, yValue, zValue: Coordinates (centroid for negative potentials, first vertex for positive)
* izquierda: Spatial ranking tag within potential group

# Acknowledgments

Based on marching cubes implementation adapted from alvin-yang68/Marching-Cubes and the table found on https://paulbourke.net/geometry/polygonise/



















