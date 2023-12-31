# explosive-immunization

Explosive immunization algorithm for network fragmentation introduced in the article Pau Clusella, Peter Grassberger,
Francisco J. Pérez-Reche, and Antonio Politi Phys. Rev. Lett. **117**, 208301.

Please, refer to the paper for details of the algorithm.

## Compiling

The codes are in the `Library` folder. Download the Library on your workspace and compile using

```
$ make -C Library
```

Upon success, the executable `exploimmun` will be created in you workspace

## Using the algorithm

To run the code use

```
$ ./exploimmun <m> <network.txt>
```

where

* `m` is a positive integer stating the number of candidates to select at each step of the algorithm. The smaller
  the `m`, the fastest is the algorithm, but a too small number might result on a bad performance. For networks with a
  million nodes, `m=1000` might be a good options.

* `network.txt` is the file where the network is stored. The first line must be only the number of nodes of the networ.
  The following lines must contain the list of edges with the format. A edge written as `i j` means that there is a link
  between node `i` and  `j`. **The proper format of the file is not checked by the code.** A exemplary file is given in
  the repository as `ER1e5.txt`.

## Outputs generated by the algorithm

All the output files are generated in the workspace

* `output_sigma1.dat` contains the outcome of using the first score as pairs of numbers `q G(q)` where `q` is the
  fraction of vaccinated nodes, and `G(q)` the relative size of the largest network component.
* `output_sigma2.dat` same, but using the second score.
* `threshold_conditions.dat` contains the list of vaccinated nodes at the approximated percolation threshold (computed
  as 1/sqrt(N)). A `1` means vaccinated, and a `0` means unvaccinated.

## Algorithm parameters

The only parameter the program requires is `m`. The effective degree cut-off is set to 6. The threshold where the second
score is invoked is set to 1/sqrt(N). These two parameters can be easily changed from the file `Library/exploimmun.c`
