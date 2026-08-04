# E011 results — do not apply a dataset-wide rewrite

The final focused repair suite passed.  The repair function can recover a
known CP1252/UTF-8 mojibake fixture and leaves clean/un-decodable text intact.

On the raw ViFinQA files, however, it adopted **zero** repairs:

| Corpus | Strings | Adopted | Marker score before → after |
| --- | ---: | ---: | --- |
| Questions | 1,012 | 0 | 0 → 0 |
| Fixed report sample | 4,000 lines from 50 reports | 0 | 144 → 144 |

The report marker count is not evidence of mojibake: a flagged example was
valid Vietnamese uppercase text (`ĐÃ ĐƯỢC`).  Therefore this marker heuristic
is not a reliable corpus-quality metric.  The safe decision is to retain raw
UTF-8 source text and not enable a global repair transform.  Terminal display
artefacts must not be mistaken for source corruption.
