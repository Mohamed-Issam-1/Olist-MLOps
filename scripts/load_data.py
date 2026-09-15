from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


# --------------------------------------------------
# Database connection
# --------------------------------------------------

DATABASE_URL = (
    "postgresql+psycopg2://"
    "olist_user:olist_password@127.0.0.1:5433/olist"
)

engine = create_engine(DATABASE_URL)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# --------------------------------------------------
# CSV file -> PostgreSQL table name
# --------------------------------------------------

FILES = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "product_category_translation",
}


def test_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database();")
        )

        print(f"Connected to database: {result.scalar()}")


def load_csv_files():

    for file_name, table_name in FILES.items():

        file_path = DATA_DIR / file_name

        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            continue

        print(f"\nReading: {file_name}")

        df = pd.read_csv(file_path)

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        print(f"Loading into PostgreSQL table: {table_name}")

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=5000,
        )

        print(f"Loaded successfully: {table_name}")


if __name__ == "__main__":

    print("Testing database connection...")
    test_connection()

    print("\nStarting Olist data ingestion...")
    load_csv_files()

    print("\nAll available files have been processed.")