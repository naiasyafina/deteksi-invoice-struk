import os

from pydantic import BaseModel

BUSY = {429, 500, 502, 503, 504}
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
RULES ="""- Dates as YYYY-MM-DD, empty string if not printed.
- items: every purchased line. price is the line total (qty x unit price).
- Money as plain numbers without thousand separators or currency symbols, dot for decimals. 0 if not printed.
- currency: ISO code such as IDR or USD.
- Copy text as printed, do not translate."""


class Item(BaseModel):
    name: str
    qty: float
    price: float


class Struk(BaseModel):
    store_name: str
    date: str
    items: list[Item]
    subtotal: float
    tax: float
    total: float
    payment_method: str
    currency: str


class Invoice(BaseModel):
    invoice_no: str
    vendor: str
    customer: str
    issue_date: str
    due_date: str
    items: list[Item]
    subtotal: float
    tax: float
    total: float
    currency: str


SCHEMAS = {"struk": Struk, "invoice": Invoice}
PROMPTS = {
    "struk": "Read this shopping receipt and extract its data.\n- store_name: store or merchant name.\n- total: final amount paid.\n- payment_method: e.g. cash, debit, QRIS, empty string if not printed.\n" + RULES,
    "invoice": "Read this invoice and extract its data.\n- vendor: the issuing company. customer: the billed party.\n- total: final amount due.\n" + RULES,
}


def make_client():
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        return None
    from google import genai
    from google.genai import types

    retry = types.HttpRetryOptions(attempts=3, initial_delay=2, max_delay=10, http_status_codes=sorted(BUSY))
    return genai.Client(http_options=types.HttpOptions(retry_options=retry))


def model_order() -> list[str]:
    return list(dict.fromkeys([os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), *FALLBACK_MODELS]))


def generate(client, **request):
    models = model_order()
    for model in models:
        try:
            return client.models.generate_content(model=model, **request)
        except Exception as error:
            if getattr(error, "code", None) not in BUSY or model == models[-1]:
                raise


def extract(client, image: bytes, mime_type: str, jenis: str) -> dict:
    from google.genai import types

    response = generate(
        client,
        contents=[types.Part.from_bytes(data=image, mime_type=mime_type), PROMPTS[jenis]],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMAS[jenis]),
    )
    if response.parsed is None:
        raise ValueError("Gemini tidak mengembalikan JSON yang valid")
    return response.parsed.model_dump()


if __name__ == "__main__":

    class Busy(Exception):
        code = 503

    class Stub:
        def __init__(self, busy: set[str]):
            self.busy, self.calls, self.models = busy, [], self

        def generate_content(self, model, **_):
            self.calls.append(model)
            if model in self.busy:
                raise Busy()
            return model

    order = model_order()
    assert generate(Stub({order[0]})) == order[1]
    stub = Stub(set(order))
    try:
        generate(stub)
        raise AssertionError("harus error kalau semua model sibuk")
    except Busy:
        assert stub.calls == order
    print("ok")
