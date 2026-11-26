import io

import pandas as pd

SHEETS = {"struk": "Struk", "invoice": "Invoice"}


def to_tables(documents: list[dict]) -> dict[str, pd.DataFrame]:
    tables = {sheet: [] for sheet in (*SHEETS.values(), "Item")}
    for doc in documents:
        fields = {key: value for key, value in doc["data"].items() if key != "items"}
        tables[SHEETS[doc["jenis"]]].append({"file": doc["file"], **fields})
        tables["Item"] += [{"file": doc["file"], "jenis": doc["jenis"], **item} for item in doc["data"]["items"]]
    return {sheet: pd.DataFrame(rows) for sheet, rows in tables.items() if rows}


def to_excel(tables: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet, index=False)
    return buffer.getvalue()


if __name__ == "__main__":
    docs = [
        {"file": "a.jpg", "jenis": "struk", "data": {"store_name": "Toko A", "total": 38500, "items": [{"name": "Es Teh", "qty": 2, "price": 10000}]}},
        {"file": "b.png", "jenis": "struk", "data": {"store_name": "Toko B", "total": 5000, "items": []}},
    ]
    tables = to_tables(docs)
    assert list(tables) == ["Struk", "Item"]
    assert tables["Struk"].to_dict("records")[1] == {"file": "b.png", "store_name": "Toko B", "total": 5000}
    assert tables["Item"].iloc[0]["name"] == "Es Teh"
    back = pd.read_excel(io.BytesIO(to_excel(tables)), sheet_name=None)
    assert list(back) == ["Struk", "Item"] and back["Struk"].shape == (2, 3)
    print("ok")
