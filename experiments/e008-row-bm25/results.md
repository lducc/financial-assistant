# E008 result — initial row baseline rejected

The initial row-centric BM25 smoke run was crash-free, but its VNM fixture
failed: ubiquitous header rows containing `2023` and `VND` outranked the actual
`Doanh thu thuần` metric row. This is a valid grounding failure, not a quality
result. E008b repairs the query representation while preserving raw evidence.

