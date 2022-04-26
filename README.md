# homework problem to do a star

Input your starting values for central pressure and temperature and the star's X, Y, and Z mass-fractions to `config.json`.

`tstep` and `pstep` allow you to search in a grid of `pnum`/`tnum` points around your starting values. The list should be `[lo,hi]`, where the grid will be done by `np.linspace(Pc0-pstep[0],Pc0+pstep[1],pnum)`. If you do not want to use either of these, then set to `"None"`. If the step variable is set to `"None"`, it does not matter what the num variable is set to.

`results.txt` will be written, which gives the final r,p,t,m,l values for each starting condition.

`outtable` determines whether or not the full structure is written to `structure.txt`. This should only really be set to `"True"` if both `tstep` and `pstep` are `"None"`.
