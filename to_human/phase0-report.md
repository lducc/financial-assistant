# ViFinQA Phase 0: first evidence

## What is now verified

The pinned public dataset is internally catalogable: 1,012 contiguous question
IDs, 1,973 unique reports, 100 companies, 2015–2025, and 146,246 literal HTML
tables. The first-pass parser found no empty literal tables and recovered page
context for every one.

## What changes the engineering plan

The main early risk is data quality and metadata, not missing tables:

- 1,794 reports carry encoding/OCR-corruption markers.
- 55 report directories do not match the common consolidated/separate/
  aggregated scope convention.
- Eight PRT explanatory reports contain no literal HTML table.
- HTML-table order is a stable internal ID, but has not yet been verified as
  the organizer's submission table-position semantics.

## Next experiment

Build the minimal catalog and deterministic metadata/row retrieval baseline,
while creating a manual proxy benchmark to measure whether metadata filtering
and future OCR repair improve evidence sufficiency rather than merely ranking
similar-looking tables.

