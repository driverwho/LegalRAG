"""SQLAlchemy engine and session setup for SQLite."""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.app.config.settings import get_settings

settings = get_settings()

# SQLite engine (synchronous)
engine = create_engine(
    f"sqlite:///{settings.SQLITE_DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def _migrate_missing_columns() -> None:
    """Add any columns defined in models but missing from existing tables."""
    inspector = inspect(engine)
    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue
        existing = {col["name"] for col in inspector.get_columns(table_name)}
        for column in table.columns:
            if column.name not in existing:
                col_type = column.type.compile(dialect=engine.dialect)
                default = ""
                if column.default is not None:
                    default = f" DEFAULT {column.default.arg!r}"
                elif column.nullable:
                    default = " DEFAULT NULL"
                with engine.begin() as conn:
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type}{default}"
                    ))


def init_db() -> None:
    """Create all database tables and migrate missing columns."""
    Base.metadata.create_all(bind=engine)
    _migrate_missing_columns()
