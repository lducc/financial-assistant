from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from compliant_docs.catalog import load_companies, load_reports
from compliant_docs.parser import parse_question
from compliant_docs.retriever import minimal_report_years, retrieve_docs


def resources():
    data = ROOT / "ViFinQA_data"
    return (
        load_companies(data / "code_stock.csv"),
        load_reports(data / "financial_statements"),
    )


def test_single_company_separate():
    companies, reports = resources()
    parsed = parse_question(
        "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu?",
        companies,
    )
    docs, _ = retrieve_docs(parsed, reports)
    assert parsed.tickers == ["VJC"]
    assert parsed.scope == "separate"
    assert docs == ["VJC_financial_statements_2018_separate"]


def test_multi_company_and_comparative_year_cover():
    companies, reports = resources()
    parsed = parse_question(
        "Trong nhóm HPG, HSG, MSR và NKG năm 2023 và 2024, công ty nào tăng nhanh nhất?",
        companies,
    )
    docs, _ = retrieve_docs(parsed, reports)
    assert parsed.tickers == ["HPG", "HSG", "MSR", "NKG"]
    assert minimal_report_years([2023, 2024]) == [2024]
    assert len(docs) == 4
    assert all("_2024_consolidated" in doc for doc in docs)


def test_year_range_expands():
    companies, _ = resources()
    parsed = parse_question("Trong giai đoạn 2020-2024 của HPG, năm nào cao nhất?", companies)
    assert parsed.years == [2020, 2021, 2022, 2023, 2024]



def test_historical_alias_groups():
    companies, _ = resources()
    cases = [
        ("Trong hai doanh nghiep Dam Phu My va Dam Ca Mau", ["DPM", "DCM"]),
        ("Trong nhom Hoa Phat, Hoa Sen va Nam Kim", ["HPG", "HSG", "NKG"]),
        ("Masan, Dai Duong va Vinamilk", ["MSN", "OGC", "VNM"]),
        ("MBBank lon hon Eximbank", ["MBB", "EIB"]),
    ]
    for question, expected in cases:
        assert set(parse_question(question, companies).tickers) == set(expected)


def test_subject_is_not_confused_with_counterparty():
    companies, _ = resources()
    parsed = parse_question(
        "Trong cac nam 2015 va 2016 cua Tong Cong ty Khi Viet Nam - CTCP, "
        "gia tri ban hang voi Tong Cong ty Dien luc Dau khi Viet Nam la bao nhieu?",
        companies,
    )
    assert parsed.tickers == ["GAS"]


def test_company_name_boundaries_do_not_match_an_binh():
    companies, _ = resources()
    parsed = parse_question(
        "Trong so Tap doan Dabaco va Masan, doanh nghiep co binh quan tai san cao nhat?",
        companies,
    )
    assert "ABB" not in parsed.tickers
    assert set(parsed.tickers) == {"DBC", "MSN"}

def test_pseudo_gt_alias_regressions():
    companies, _ = resources()
    cases = [
        ("Loi nhuan sau thue cua CTCP Chung khoan FPT nam 2023", {"FTS"}),
        (
            "Chenh lech gia tri hop ly khoan dau tu vao Ngan hang TMCP "
            "Ngoai Thuong Viet Nam cua GEE nam 2025",
            {"GEE"},
        ),
        (
            "Co bao nhieu trong so cac ngan hang TMCP A Chau, TMCP Quan doi, "
            "TMCP Xuat nhap khau Viet Nam va TMCP Dau tu va Phat trien Viet Nam",
            {"ACB", "MBB", "EIB", "BID"},
        ),
    ]
    for question, expected in cases:
        assert set(parse_question(question, companies).tickers) == expected


def test_metric_wording_does_not_hide_subject_company():
    companies, _ = resources()
    cases = [
        (
            "Chenh lech giua gia goc khoan dau tu vao cong ty lien ket cua "
            "cong ty me CTCP Dien Gia Lai va cong ty me CTCP Thuy dien Da Nhim "
            "Ham Thuan Da Mi nam 2023",
            {"GEG", "DNH"},
        ),
        (
            "Gia tri thu nhap tu mua ban chung khoan dau tu cua cong ty me "
            "Ngan hang TMCP Xuat nhap khau Viet Nam tru di gia tri cua "
            "Ngan hang TMCP A Chau nam 2025",
            {"EIB", "ACB"},
        ),
    ]
    for question, expected in cases:
        assert set(parse_question(question, companies).tickers) == expected


def test_mixed_scope_full_year_coverage():
    companies, reports = resources()
    parsed = parse_question(
        "Trung binh so du tai DTK qua bao cao cong ty me cac nam 2022, 2023 "
        "va bao cao hop nhat cac nam 2024, 2025",
        companies,
    )
    docs, _ = retrieve_docs(parsed, reports, use_comparative_cover=False)
    assert parsed.scope_by_year == {
        2022: "separate",
        2023: "separate",
        2024: "consolidated",
        2025: "consolidated",
    }
    assert docs == [
        "DTK_financial_statements_2022_separate",
        "DTK_financial_statements_2023_separate",
        "DTK_financial_statements_2024_consolidated",
        "DTK_financial_statements_2025_consolidated",
    ]
