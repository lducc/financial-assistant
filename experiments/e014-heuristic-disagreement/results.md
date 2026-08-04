# E014 results — 15 high-value human-review cases

All 120 frozen E003 records joined to E009 plans, and the audit test passed.
The two independently written heuristics agree on 105 records and disagree on
15.  This is a consistency diagnostic, not operation accuracy.

The most important disagreement patterns are:

| Earlier E003 hint → E009 template | Cases |
| --- | ---: |
| lookup → aggregate | 5 |
| difference → growth/change | 5 |
| lookup → growth/change | 3 |
| selector → ratio/percent | 2 |

Five disagreement IDs are already in the E012 double-review subset: 380, 423,
427, 595, and 898.  The remaining candidates (627, 642, 643, 649, 754, 793,
920, 963, 979, 990) should be prioritized if expanding manual annotation.
