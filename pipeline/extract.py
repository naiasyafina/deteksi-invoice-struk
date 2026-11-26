import re
from datetime import date

from pipeline.postprocess import parse_money

PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")
COUNT = re.compile(r"\(?\s*(qty|items?|pcs)\s*[:=]?\s*\d+(?:[.,]\d+)?\s*\)?|\d+(?:[.,]\d+)?\s*(items?|pcs)\b", re.I)
TOTAL = re.compile(r"grand\s*total|t[o0]tal|jumlah|tagihan", re.I)
NOT_TOTAL = re.compile(r"sub|tax|pajak|ppn|disc|change|kembali|service|round", re.I)
SUBTOTAL = re.compile(r"sub\s*-?\s*t[o0]?ta?l", re.I)
TAX = re.compile(r"\b(tax|pajak|ppn|vat)\b|\bpb\s*1\b", re.I)
NOT_STORE = re.compile(r"\b(jl|jln|jalan|telp|tel|phone|hp|npwp|no|kasir|cashier|table|meja|date|tanggal|tgl|www|http|struk|receipt)\b", re.I)
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "may": 5, "jun": 6, "jul": 7,
    "agu": 8, "agt": 8, "aug": 8, "sep": 9, "okt": 10, "oct": 10, "nov": 11, "des": 12, "dec": 12,
}
NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{2,4})(?!\d)")
TEXT_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*-?\s*(" + "|".join(MONTHS) + r")[a-z]*\.?\s*-?\s*(\d{2,4})(?!\d)", re.I)


def amount(line: str) -> int | None:
    return parse_money(COUNT.sub("", PERCENT.sub("", line)))


def amount_near(lines: list[str], index: int) -> int | None:
    value = amount(lines[index])
    for neighbor in (index + 1, index - 1):
        if value is None and 0 <= neighbor < len(lines) and not re.search(r"[a-z]{3}", lines[neighbor], re.I):
            value = amount(lines[neighbor])
    return value


def find_total(lines: list[str]) -> int | None:
    def values(pattern: re.Pattern, exclude: re.Pattern | None = None) -> list[int]:
        found = [amount_near(lines, i) for i, line in enumerate(lines) if pattern.search(line) and not (exclude and exclude.search(line))]
        return [value for value in found if value and value > 0]

    return max(values(re.compile(r"grand\s*total", re.I)) or values(TOTAL, NOT_TOTAL), default=None)


def find_labeled(lines: list[str], pattern: re.Pattern) -> int | None:
    for i, line in enumerate(lines):
        if pattern.search(line):
            value = amount_near(lines, i)
            if value is not None:
                return abs(value)
    return None


def to_date(day: int, month: int, year: int) -> str | None:
    if year < 100:
        year += 2000
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def find_date(lines: list[str]) -> str | None:
    for line in lines:
        for a, b, c in NUMERIC_DATE.findall(line):
            parsed = to_date(int(c), int(b), int(a)) if len(a) == 4 else to_date(int(a), int(b), int(c))
            if parsed:
                return parsed
        for day, month, year in TEXT_DATE.findall(line):
            parsed = to_date(int(day), MONTHS[month[:3].lower()], int(year))
            if parsed:
                return parsed
    return None


def find_store(lines: list[str]) -> str | None:
    for line in lines[:5]:
        letters = sum(char.isalpha() for char in line)
        if letters >= 3 and letters / max(len(line.replace(" ", "")), 1) >= 0.6 and not NOT_STORE.search(line):
            return line.strip()
    return None


def find_items(lines: list[str]) -> list[dict]:
    # ponytail: baris berangka sebelum subtotal/total dianggap item; item multi-baris dan diskon per item tidak ditangani
    items = []
    for line in lines:
        if SUBTOTAL.search(line) or TOTAL.search(line) or TAX.search(line):
            break
        price = amount(line)
        name = " ".join(word for word in line.split() if re.search(r"[a-z]", word, re.I) and word.lower() not in {"x", "@", "rp"})
        if price and price >= 100 and len(name) >= 2 and not NOT_STORE.search(line):
            items.append({"name": name, "price": price})
    return items


def extract_regex(lines: list[str]) -> dict:
    return {
        "items": find_items(lines),
        "store_name": find_store(lines),
        "date": find_date(lines),
        "subtotal": find_labeled(lines, SUBTOTAL),
        "tax": find_labeled(lines, TAX),
        "total": find_total(lines),
    }


if __name__ == "__main__":
    receipt = [
        "WARUNG BU SRI",
        "Jl. Imam Bonjol No. 5",
        "Tgl: 03/11/2026 12:30",
        "Es Teh 2 x 5.000 10.000",
        "Nasi Goreng 25.000",
        "Sub Total 35.000",
        "PB1 10% 3.500",
        "TOTAL",
        "38.500",
        "Cash 50.000",
        "Kembali 11.500",
    ]
    assert extract_regex(receipt) == {
        "items": [{"name": "Es Teh", "price": 10000}, {"name": "Nasi Goreng", "price": 25000}],
        "store_name": "WARUNG BU SRI",
        "date": "2026-11-03",
        "subtotal": 35000,
        "tax": 3500,
        "total": 38500,
    }, extract_regex(receipt)
    assert find_date(["12 Agustus 2025"]) == "2025-08-12"
    assert find_date(["2024-02-30", "2024-02-29"]) == "2024-02-29"
    assert find_date(["Rp 1.250.000"]) is None
    assert find_total(["Total Item 3", "Grand Total Rp 120.000", "Total 100.000"]) == 120000
    assert find_total(["TOTAL (Qty 2.00) 60.000"]) == 60000
    assert find_total(["3 ITEMS TOTAL 498,608"]) == 498608
    assert find_total(["TOTAL QTY 2 ITEM 2 PCS", "Total 22.000"]) == 22000
    assert find_total(["145,900", "GRAND TOTAL", "CASH 200,000"]) == 145900
    assert find_labeled(["PB 1 10% 8.100"], TAX) == 8100
    print("ok")
