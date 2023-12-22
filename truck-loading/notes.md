# Benchmark on OR Tools - CP sat

## TODO

- [x] Change definition of $c_{ij}$: instead of having the second dimension (columns) equal to n_items \* n_trucks, define beforehand the maximum number of trucks to be used (observing some known solutions).

> `n_trucks` = [a, b, c], with a = n. of trucks of type 0, b = n. of trucks of type 1, ...

- [x] Add 3rd dimension in $c_{i,j}$ - the length of the 3rd dimension is given by the corresponding value in vector `n_trucks`


---

- [ ] Clean code & refactoring
- [ ] Translate item format
- [ ] Figure out how to consider orientation
