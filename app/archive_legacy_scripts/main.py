from src.database import DatabaseManager


def main() -> None:
    db = DatabaseManager()
    db.create_tables()

    sample_batch = [
        ("000001.SZ", "2024-01-01 09:30:00", "5min", 10, 10.5, 9.8, 10.2, 150000),
        ("000001.SZ", "2024-01-01 09:35:00", "5min", 10.2, 10.6, 10.0, 10.4, 120000),
        ("000001.SZ", "2024-01-01 09:40:00", "5min", 10.4, 10.7, 10.3, 10.5, 110000),
    ]

    db.insert_kline_batch(sample_batch)

    rows = db.fetch_by_stock("000001.SZ", "5min")

    print("Query result:")
    for row in rows:
        print(row)

    db.close()


if __name__ == "__main__":
    main()
