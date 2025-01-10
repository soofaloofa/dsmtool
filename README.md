# dsmtool

`dsmtool` provides command line utilities for interacting with a [Design Structure Matrix](https://dsmweb.org).

## Installation

## Usage

At any time, you can access help for the tool using the `--help` flag.

```
dsmtool --help
```

### Clustering

To run the clustering algorithm, specify the input DSM and desired output DSM after clustering:

```
dsmtool cluster -i <input dsm> -o <output dsm>
```

To output the cost history, add the `-c` flag. Cost history will be output to
`cost_history.png`.

#### Clustering Algorithm Support

The tool currently supports a single clustering algorithm using a
simplified simulated annealing method. This algorithm is a modified version of the original [Matlab
code](https://dsmweb.org/matlab-macro-for-clustering-dsms/) developed by Ronnie
Thebeau as part of his [master's
thesis](https://dsmweb.org/wp-content/uploads/2019/05/msc_thebeau.pdf).

For more on the clustering algorithm, see [Clustering a DSM Using Simulated
Annealing](https://sookocheff.com/post/dsm/clustering-a-dsm-using-simulated-annealing).

### DSM File Format

A Design Structure Matrix (DSM) file is CSV file that contains an N times N
matrix of values. Each line of the file contains a row of the matrix. Within the
line, values are separated by commas, and leading and trailing white space
around each value is discarded. Both integer and real values are supported, such
as 0, 1, 1.5, 1e3 and 1.5e-4. Negative values, NaN and infinite values are not
allowed. Before the first number at each row there should be a label indicating
the name of the element of that row. Above the first line of data
the same labeling scheme should exist as well. 

The following example shows a sample DSM file that could be used as input:

```csv
 ,A,B,C,D,E,F,G
A,1,0,0,0,0,1,0
B,1,1,1,1,0,0,1
C,0,0,1,1,0,0,0
D,0,1,1,1,1,0,0
E,1,0,0,1,1,1,0
F,1,0,0,0,1,1,0
G,0,1,1,1,0,0,1
```

## Reference Algorithms

* Original hill climbing variant: https://dspace.mit.edu/handle/1721.1/29168
* Improved hill climbing variant: https://www.researchgate.net/publication/267489785_Improved_Clustering_Algorithm_for_Design_Structure_Matrix
* Markov version: https://gitlab.eclipse.org/eclipse/escet/escet/-/blob/develop/common/org.eclipse.escet.common.dsm/src/org/eclipse/escet/common/dsm/DsmClustering.java