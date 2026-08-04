# E011 — Conservative Vietnamese mojibake repair

## Question

Can a reversible byte-round-trip repair recover readable Vietnamese from the
dataset's UTF-8-decoded-as-Latin-1 artefacts without corrupting already valid
OCR text?

## Method

1. Keep every raw string unchanged as the authoritative source.
2. Generate a candidate only by `latin-1` encoding followed by UTF-8 decoding.
3. Adopt the candidate only if a deterministic mojibake-marker score strictly
   decreases.  If the transform cannot encode/decode, retain raw text.
4. Test known corrupted, clean Vietnamese, ASCII, and unrepairable inputs.
5. Measure repair/adoption counts on the 1,012 questions and a deterministic
   sample of report text; do not claim OCR correctness from marker reduction.

## Acceptance criteria

- Clean text and non-transformable text are byte-for-byte preserved.
- Known `Tá»•ng ... Ä‘á»“ng` text is repaired to Vietnamese.
- The full audit reports raw and repaired marker counts separately.

## Decision boundary

Use repaired text only as a parallel retrieval/planning view; report line
addresses, HTML and raw evidence cells always remain tied to the original file.
