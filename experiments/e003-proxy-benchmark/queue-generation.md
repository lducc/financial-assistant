# E003 queue-generation record

The first queue-generation implementation was rejected before annotation: its
lexical ordering made all 120 selected examples `compositional`, violating the
protocol's intended complexity coverage. No labels or model results were
generated from that queue.

The replacement sampler is deterministic and round-robins full
operation × complexity × scope × unit strata. Its feature values are question
parser hints, not gold labels; annotation must correct them where necessary.

