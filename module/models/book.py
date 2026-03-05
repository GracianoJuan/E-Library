from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from database import Base


class Book(Base):
    __tablename__ = 'books'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    description = Column(String)
    author = Column(String, index=True)
    total_likes = Column(Integer, default=0)

    categories = relationship(
        "Category",
        secondary="book_category",
        back_populates="books"
    )

    likes = relationship("Like", back_populates="book")
