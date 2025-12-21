from typing import Optional
from datetime import datetime

from sqlalchemy import create_engine, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session
DATABASE_URL = "sqlite:///./history.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class RequestHistory(Base):
    __tablename__ = "request_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    endpoint: Mapped[Optional[str]] = mapped_column(String, index=True)
    method: Mapped[Optional[str]] = mapped_column(String, index=True)
    body: Mapped[Optional[str]] = mapped_column(Text)
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    input_size: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
