# Direction check using the supplied ideation workflow

The blueprint already specifies a clear problem-first question, so the
brainstorming workflow is used here to test its assumptions rather than replace
the project with generic idea generation.

## Candidate hypotheses from tension and failure analysis

| Candidate | Problem-first value | Feasibility now | Decision |
| --- | --- | --- | --- |
| H1 metadata-first filtering | avoids near-duplicate report confusion | high | test in baseline |
| H2 row-centric multi-view retrieval | reduces wrong-row grounding | medium | defer until proxy labels |
| H3 role decomposition | addresses multi-table closure | medium | defer until benchmark |
| H4 coverage-aware set selection | treats sufficiency as a set property | medium | defer until H3 |
| H5 ontology weak supervision | supplies constraints without labels | medium | retain as main research direction |
| H6 typed IR compilation | reduces arbitrary-code failures | high | test after evidence catalog |
| H7 unit verification | prevents executable but wrong answers | high | test after compiler |
| H8 accounting identities | detects silent OCR errors | medium | retain as verifier ablation |
| H9 targeted repair | avoids wholesale regeneration | medium | defer until taxonomy data |
| H10 cross-lingual transfer | supplies program supervision | low before proxy | park |
| H11 confidence routing | reduces compute variance | medium | defer until reliable paths exist |
| H12 minimal evidence | tests irrelevant-context harm | medium | pair with H4 |

## Two-sentence selection

Vietnamese financial table QA fails because evidence is duplicated, OCR is
noisy, and the public competition supplies no retrieval or program labels.
We will first measure a deterministic metadata-and-catalog baseline, then test
whether accounting constraints yield a smaller sufficient evidence set and a
typed, verifiable program path.

## Strongest objection and response

**Objection:** accounting constraints could simply reject legitimate special
cases and lower recall. **Response:** treat them as logged validators and
ablate them against a frozen, stratified proxy set; they never silently modify
source values.

