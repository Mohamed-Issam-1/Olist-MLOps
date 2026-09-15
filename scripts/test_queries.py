from sqlalchemy import create_engine, text


DATABASE_URL = (
    "postgresql+psycopg2://"
    "olist_user:olist_password@127.0.0.1:5433/olist"
)

engine = create_engine(DATABASE_URL)


def check_tables():
    query = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)

    with engine.connect() as connection:
        result = connection.execute(query)

        tables = [row[0] for row in result]

    print("\nTables found:")
    for table in tables:
        print("-", table)


def check_row_counts():
    tables = [
        "customers",
        "geolocation",
        "order_items",
        "order_payments",
        "order_reviews",
        "orders",
        "products",
        "sellers",
        "product_category_translation",
    ]

    print("\nRow counts:")

    with engine.connect() as connection:

        for table in tables:
            query = text(f"SELECT COUNT(*) FROM {table};")
            result = connection.execute(query)

            count = result.scalar()

            print(f"{table}: {count:,}")


def test_join():
    query = text("""
        SELECT
            o.order_id,
            o.order_status,
            c.customer_city,
            c.customer_state
        FROM orders AS o
        JOIN customers AS c
            ON o.customer_id = c.customer_id
        LIMIT 5;
    """)

    with engine.connect() as connection:
        result = connection.execute(query)

        rows = result.fetchall()

    print("\nJOIN test: orders + customers")

    for row in rows:
        print(row)


if __name__ == "__main__":

    print("Testing Olist PostgreSQL database...")

    check_tables()
    check_row_counts()
    test_join()

    print("\nDatabase tests completed successfully.")