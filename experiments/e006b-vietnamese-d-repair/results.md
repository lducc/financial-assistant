# E006b result — passed

After explicitly folding Vietnamese `đ` to `d` after decomposition, all curated
fixtures pass: dot/comma thousands and decimals, parentheses negatives, missing
dashes, ambiguous separators, percentage points, `triệu đồng`, and `nghìn tỷ
đồng`.

The parser preserves Decimal precision and keeps percentage values in percentage
points. This clears the curated-fixture gate for table normalization, but not a
corpus-wide accuracy claim.

