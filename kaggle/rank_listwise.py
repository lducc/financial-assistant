"""Kaggle GPU notebook: reorder candidate windows with Qwen3-8B, listwise.

The pointwise reranker asks "does this table contain the line item, yes or no"
about each candidate separately. Two tables in the same report that both carry
the item both answer yes at near 1.0, and nothing in the scoring function ever
compares them — which is the measured failure. Of the benchmark questions whose
top-1 is not gold, the median score gap between the wrong pick and the best gold
is 0.067, 47% are within 0.05, and every one of those misses is intra-report.

This shows the model 20 candidates at once and asks for an order. Oracle
reordering of the top 20 of the current ranking reaches F2 0.7893 against 0.6562
today, so the headroom is real; whether a model can find it is what this run
measures.

Reads windows.jsonl from export_listwise_windows.py and writes orders.jsonl for
apply_listwise_order.py to splice locally. Retrieval is untouched: only the order
of already-retrieved candidates changes, so a bad run is one file to delete.

Paths and the model read from the environment, so the same file runs on a rented
GPU. Orders append per question and finished IDs are skipped, so a session that
hits the 12 h limit is rerun rather than restarted.
"""

import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Instruct model, not the reranker: this task needs generation, not a yes/no
# logit. Same family that works on this data, released well before the
# 2026-06-01 cutoff, and 8.2 GB in int8 so a T4 has room. The 14B cap is per
# model, so Qwen3-14B is legal and is the upgrade if 8B ranks well but not well
# enough; it is roughly 1.7x slower and tight on a 16 GB card.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
WINDOWS_PATH = os.environ.get(
    "WINDOWS_PATH", "/kaggle/input/vifinqa-rerank-pairs/windows_bench.jsonl",
)
ORDERS_PATH = os.environ.get("ORDERS_PATH", "/kaggle/working/orders.jsonl")
RESUME_PATH = os.environ.get("RESUME_PATH")
# Windows measure ~1,643 tokens; 3072 leaves room for the longest without
# truncating a candidate out of the comparison.
MAX_LENGTH = 3072
# A permutation of 20 labels is ~60 tokens. Cutting generation short costs only
# the tail of the order, which the parser fills in from the incoming ranking.
MAX_NEW_TOKENS = 160

INSTRUCTION = (
    "Bạn xếp hạng các bảng trong báo cáo tài chính. Mọi bảng dưới đây đều thuộc "
    "đúng công ty, đúng kỳ và đúng loại báo cáo, nên những yếu tố đó không phân "
    "biệt được chúng. Hãy xếp theo mức độ bảng đó BÁO CÁO chỉ tiêu được hỏi: "
    "bảng có chỉ tiêu là một dòng riêng kèm giá trị của kỳ được hỏi xếp trước. "
    "Nhãn trùng chữ nhưng chỉ nhắc tới chỉ tiêu — danh sách công ty con, bên "
    "liên quan, bảng biến động hay phân bổ, dòng chỉ là tiêu đề cột — xếp sau. "
    "Thuyết minh hoặc bảng chi tiết nêu lại đúng con số vẫn được tính là báo cáo "
    "chỉ tiêu đó."
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def already_done(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def build_prompt(record, tokenizer):
    items = "; ".join(record.get("line_items") or [])
    header = f"Câu hỏi: {record['question']}"
    if items:
        header += f"\nChỉ tiêu cần tìm: {items}"
    blocks = "\n\n".join(
        f"[{position}] {candidate['text']}"
        for position, candidate in enumerate(record["candidates"], 1)
    )
    count = len(record["candidates"])
    task = (
        f"\n\nHãy xếp {count} bảng trên theo thứ tự giảm dần mức độ báo cáo chỉ tiêu "
        f"được hỏi. Chỉ trả lời bằng các nhãn, ví dụ: [2] > [5] > [1] > ... "
        f"Phải liệt kê đủ cả {count} nhãn, không giải thích."
    )
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"{header}\n\n{blocks}{task}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


LABEL = re.compile(r"\[(\d+)\]")


def parse_permutation(text, size):
    """Zero-based positions named by [n] labels, first mention winning.

    Kept identical to src/vifinqa/listwise.py, which is the tested copy. A
    generated permutation can repeat a label, invent one out of range or stop
    early, and none of those may lose a table — the caller appends whatever went
    unmentioned, so a garbled generation degrades to the incoming order.
    """
    seen = {}
    for match in LABEL.finditer(text or ""):
        position = int(match.group(1)) - 1
        if 0 <= position < size:
            seen.setdefault(position, None)
    return list(seen)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    print(f"{MODEL_NAME} loaded in 8-bit, max_length={MAX_LENGTH}", flush=True)

    records = load_jsonl(WINDOWS_PATH)
    done = already_done(ORDERS_PATH) | already_done(RESUME_PATH)
    if done:
        print(f"resuming, {len(done)} questions already ordered", flush=True)
    pending = [record for record in records if record["id"] not in done]
    print(f"{len(pending)} questions to order", flush=True)

    started, complete = time.time(), 0
    with open(ORDERS_PATH, "a", encoding="utf-8") as out:
        for number, record in enumerate(pending, 1):
            prompt = build_prompt(record, tokenizer)
            encoded = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            positions = parse_permutation(text, len(record["candidates"]))
            complete += len(positions) == len(record["candidates"])
            out.write(json.dumps({
                "id": record["id"],
                # Table IDs, not positions: the local side must not depend on the
                # window being rebuilt identically.
                "order": [record["candidates"][position]["table_id"] for position in positions],
                "raw": text.strip()[:200],
            }, ensure_ascii=False) + "\n")
            out.flush()
            if number % 25 == 0:
                rate = number / (time.time() - started)
                print(
                    f"{number}/{len(pending)} questions, {rate * 60:.1f}/min, "
                    f"eta {(len(pending) - number) / rate / 60:.0f} min, "
                    f"full permutations {complete}/{number}",
                    flush=True,
                )

    print(
        f"wrote {ORDERS_PATH} in {(time.time() - started) / 60:.1f} min; "
        f"{complete}/{len(pending)} returned a complete permutation",
        flush=True,
    )


if __name__ == "__main__":
    main()
