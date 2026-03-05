from sqlalchemy import Table, Column, Integer, ForeignKey
from ...database import Base

book_category = Table(
    'book_category',
    Base.metadata,
    Column('book_id', Integer, ForeignKey('books.id'), primary_key=True),
    Column('category_id', Integer, ForeignKey('categories.id'), primary_key=True),
)
