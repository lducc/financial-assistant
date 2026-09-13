# Byte-identical to the prompt in kaggle/rerank_qwen_8b.py; a test pins the two.
INSTRUCTION = (
    "The Query asks about a Vietnamese listed company's financial statements. "
    "The Document is one table from a filing whose first line identifies its "
    "ticker, reporting period, and scope (hợp nhất or công ty mẹ). The candidate "
    "pool is already narrowed to compatible filings, but reject a Document when "
    "that filing line conflicts with the Query. Answer yes only when the table "
    "reports an exact requested accounting item with a value for the requested "
    "period. For a multi-step question, a table is yes when it supplies any one "
    "required input. Do not accept a merely related label, a reference to an "
    "item, a related-party or subsidiary listing, a movement or allocation "
    "schedule, or a column header. Prefer the statement or note that reports the "
    "figure; a note or segment breakdown that validly restates it is yes."
)

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)

SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rerank_prompt(query: str, document: str) -> str:
    return (
        f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}"
        f"\n<Document>: {document}{SUFFIX}"
    )
