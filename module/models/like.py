from datetime import datetime
from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base


class Like(Base):
    __tablename__ = 'likes'

    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    book_id = Column(Integer, ForeignKey('books.id'), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="likes")
    book = relationship("Book", back_populates="likes")
