import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

PROMPT = """Read this receipt and extract its data.
- store: store or merchant name as printed.
- purchase_date: YYYY-MM-DD, empty string if not printed.
- items: every purchased line. price is the line total (qty x unit price).
- discount: total discount, 0 if none.
- tax: tax or service charge added on top of the item prices, 0 if already included or not printed.
- total: final amount paid.
Write all money as plain integers in rupiah, without separators or currency symbols."""


class Item(BaseModel):
    name: str
    qty: int
    price: int


class Receipt(BaseModel):
    store: str
    purchase_date: str
    items: list[Item]
    discount: int
    tax: int
    total: int


def read_receipt(image_bytes: bytes, mime_type: str) -> Receipt | None:
    client = genai.Client()
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Receipt,
        ),
    )
    return response.parsed


def expected_total(receipt: Receipt) -> int:
    return sum(item.price for item in receipt.items) - receipt.discount + receipt.tax
