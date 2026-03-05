from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

# --- user schemas ---
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

# --- category schemas ---
class CategoryBase(BaseModel):
    name: str

class CategoryCreate(CategoryBase):
    pass

class CategoryOut(CategoryBase):
    id: int

    class Config:
        orm_mode = True

# --- book schemas ---
class BookBase(BaseModel):
    title: str
    description: Optional[str] = None
    author: Optional[str] = None

class BookCreate(BookBase):
    category_ids: Optional[List[int]] = []

class BookOut(BookBase):
    id: int
    total_likes: int
    categories: List[CategoryOut] = []

    class Config:
        orm_mode = True

# --- like schemas ---
class LikeBase(BaseModel):
    user_id: int
    book_id: int

class LikeOut(LikeBase):
    timestamp: datetime

    class Config:
        orm_mode = True

# --- auth schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
