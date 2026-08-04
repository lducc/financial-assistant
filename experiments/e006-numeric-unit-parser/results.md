# E006 result — initial parser rejected

The first curated fixture run exposed a normalization failure: Unicode
decomposition preserves Vietnamese `đ`, so headers such as `triệu đồng` were
not recognized by the unit parser. The numeric-separator fixtures themselves
passed. This result is retained and repaired by E006b.

