Implementation of the K-Means algorithm in Python with GUI written in PyGame.

Includes code to generate:
1) The Delaunay Triangulation of a set of points
1.a) Algorithm used: brute-force.
1.b) Runtime: O(n^4) - terrible but good enough for size of inputs used.
2) The Voronoi Tessellation of a set of points
2.a) Algorithm used:
2.a.i) Compute delaunay triangulation
2.a.ii) Get dual graph of triangulation (custom cowboy algorithm)

Try: python k_means.py --help
