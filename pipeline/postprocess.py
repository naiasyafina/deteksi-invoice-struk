import re

NUMBER = re.compile(r"-?\d[\d.,]*")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_money(text: str) -> int | None:
    tokens = NUMBER.findall(text)
    if not tokens:
        return None
    token = tokens[-1].rstrip(".,")
    negative = token.startswith("-")
    if re.search(r"[.,]\d{2}$", token):
        token = token[:-3]
    digits = re.sub(r"\D", "", token)
    if not digits:
        return None
    return -int(digits) if negative else int(digits)


if __name__ == "__main__":
    cases = {
        "Rp 12.500": 12500,
        "Rp12.500,-": 12500,
        "12.500,00": 12500,
        "12,500": 12500,
        "12,500.00": 12500,
        "1.250.000": 1250000,
        "TOTAL 45.000": 45000,
        "2 x 12.500": 12500,
        "-5,000": -5000,
        "500": 500,
        "Rp": None,
    }
    for text, expected in cases.items():
        assert parse_money(text) == expected, (text, parse_money(text))
    assert normalize("Es Teh (L)") == "estehl"
    print("ok")
