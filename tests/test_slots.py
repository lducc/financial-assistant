from vifinqa.slots import append_confidence_item_tables, named_line_items


def test_named_line_items_keeps_longest_explicit_metric():
    labels = ("doanh thu thuan", "doanh thu", "nam 2024", "cong ty co phan")

    assert named_line_items(
        "Doanh thu thuần của Công ty Cổ phần ABC năm 2024 là bao nhiêu?", labels,
    ) == ["doanh thu thuan"]


def test_append_adds_confident_item_carriers_only():
    record = {
        "selected_docs": ["R1"],
        "candidates": [
            {"table_id": "R1|1", "text": "Doanh thu"},
            {"table_id": "R1|2", "text": "Chi phí"},
            {"table_id": "R1|3", "text": "Doanh thu phụ"},
            {"table_id": "R1|4", "text": "Doanh thu khác"},
        ],
    }
    ranking = append_confidence_item_tables(
        record, ["R1|1"], ["R1|1", "R1|2", "R1|3", "R1|4"], ["doanh thu"],
        {"R1|1": 0.9, "R1|2": 0.8, "R1|3": 0.7, "R1|4": 0.0001}, logit_gap=4,
    )

    assert ranking == ["R1|1", "R1|3"]
