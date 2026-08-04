# E006b — Vietnamese `đ` normalization repair

**Type:** confirmatory targeted repair of E006  
**Hypothesis:** Explicitly folding `đ` to `d` after Unicode decomposition will
restore recognition of Vietnamese currency headers without changing numeric
separator behavior.

## Decision rule

The original E006 fixtures must pass, including `triệu đồng` and `nghìn tỷ
đồng`; no previously passing numeric fixture may regress.

