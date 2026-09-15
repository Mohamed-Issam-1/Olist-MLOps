from sqlalchemy import create_engine, text

DATABASE_URL = (
    "postgresql+psycopg2://"
    "olist_user:olist_password@127.0.0.1:5433/olist"
)

engine = create_engine(DATABASE_URL)

with engine.connect() as connection:
    result = connection.execute(text("SELECT current_database();"))
    print("Connected to database:", result.scalar())