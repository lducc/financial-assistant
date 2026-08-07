from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from .catalog import Company
from .parser import ParsedQuestion


SYSTEM_PROMPT = """You are a high-recall metadata retriever for Vietnamese financial reports, NOT an answer ranker.
Select only from the supplied ticker allow-list. Return EVERY subject company whose own financial statements must be read before filtering, comparison, ranking, or calculation. Do not guess which company wins.
Do not select a company that is only a customer, supplier, investee, or transaction counterparty inside the requested metric.
Historical names in the official reports include: Dam Phu My=DPM; Dam Ca Mau=DCM; Vinamilk=VNM; Dabaco=DBC; Sao Mai=ASM; Nam Kim=NKG; Sai Gon Thuong Tin=STB; Eximbank=EIB; MBBank=MBB; Tap doan Dat Xanh=DXG; Dich vu Bat dong san Dat Xanh=DXS.
Examples:
- 'Trong nhom Hoa Phat, Hoa Sen va Nam Kim ...' -> HPG,HSG,NKG.
- 'Trong hai doanh nghiep Dam Phu My va Dam Ca Mau ...' -> DPM,DCM.
- 'Masan, Dai Duong va Vinamilk ...' -> MSN,OGC,VNM.
- 'Gia tri ban hang cua Tong Cong ty Khi Viet Nam voi Tong Cong ty Dien luc Dau khi ...' -> GAS only, because POW is the counterparty.
Return strict JSON with exactly these keys:
{"tickers":["ABC"],"confidence":0.95,"reason":"short reason"}
Never invent a ticker."""

def prompt_for(question: str, parsed: ParsedQuestion, companies: dict[str, Company]) -> str:
    candidates = [
        {"ticker": ticker, "official_name": companies[ticker].name}
        for ticker in sorted(companies)
    ]
    return (
        "Question:\n" + question + "\n\nAllowed candidates:\n" +
        json.dumps(candidates, ensure_ascii=False) +
        "\n\nIdentify all subject companies that provide the financial statements needed to answer."
    )


class OllamaResolver:
    def __init__(self, model: str, cache_path: Path, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.cache_path = cache_path
        self.base_url = base_url.rstrip("/")
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        return {
            row["key"]: row
            for row in (json.loads(line) for line in self.cache_path.read_text("utf-8").splitlines() if line.strip())
        }

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self.cache[key] for key in sorted(self.cache)]
        self.cache_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", "utf-8")

    def resolve(self, question_id: int, question: str, parsed: ParsedQuestion, companies: dict[str, Company]) -> dict[str, Any]:
        user_prompt = prompt_for(question, parsed, companies)
        key = hashlib.sha256((self.model + SYSTEM_PROMPT + user_prompt).encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 2026, "num_predict": 300},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=240)
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"tickers": [], "confidence": 0.0, "reason": "invalid_json"}
        allowed = set(companies)
        tickers = [str(value).upper() for value in result.get("tickers", [])]
        tickers = list(dict.fromkeys(value for value in tickers if value in allowed))
        record = {
            "key": key,
            "question_id": question_id,
            "model": self.model,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "user_prompt": user_prompt,
            "raw_response": raw,
            "tickers": tickers,
            "confidence": float(result.get("confidence", 0.0)),
        }
        self.cache[key] = record
        self._save()
        return record

