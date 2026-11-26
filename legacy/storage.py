import sqlite3

import pandas as pd

from receipt_reader import Receipt

DB_PATH = "receipts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY,
    store TEXT NOT NULL,
    purchase_date TEXT NOT NULL,
    discount INTEGER NOT NULL,
    tax INTEGER NOT NULL,
    total INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    receipt_id INTEGER NOT NULL REFERENCES receipts(id),
    name TEXT NOT NULL,
    qty INTEGER NOT NULL,
    price INTEGER NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def is_duplicate(receipt: Receipt) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM receipts WHERE lower(store) = lower(?) AND purchase_date = ? AND total = ?",
            (receipt.store, receipt.purchase_date, receipt.total),
        ).fetchone()
    return row is not None


def save_receipt(receipt: Receipt) -> int:
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO receipts (store, purchase_date, discount, tax, total) VALUES (?, ?, ?, ?, ?)",
            (receipt.store, receipt.purchase_date, receipt.discount, receipt.tax, receipt.total),
        )
        receipt_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO items (receipt_id, name, qty, price) VALUES (?, ?, ?, ?)",
            [(receipt_id, item.name, item.qty, item.price) for item in receipt.items],
        )
    return receipt_id


def delete_receipt(receipt_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM items WHERE receipt_id = ?", (receipt_id,))
        conn.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def load_receipts() -> pd.DataFrame:
    return query(
        "SELECT id, store, purchase_date, discount, tax, total FROM receipts ORDER BY purchase_date DESC, id DESC"
    )


def load_items(receipt_id: int) -> pd.DataFrame:
    return query("SELECT name, qty, price FROM items WHERE receipt_id = ?", (receipt_id,))


def monthly_totals() -> pd.DataFrame:
    return query(
        "SELECT substr(purchase_date, 1, 7) AS month, SUM(total) AS spent FROM receipts GROUP BY month ORDER BY month"
    )


def store_totals() -> pd.DataFrame:
    return query(
        "SELECT store, COUNT(*) AS visits, SUM(total) AS spent FROM receipts GROUP BY lower(store) ORDER BY spent DESC"
    )
