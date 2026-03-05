from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas, deps

router = APIRouter(
    prefix="/books",
    tags=["books"]
)

@router.post("/", response_model=schemas.BookOut)
def create_book(book_in: schemas.BookCreate,
                db: Session = Depends(deps.get_db),
                current_user: models.User = Depends(deps.get_current_user)):
    book = models.Book(
        title=book_in.title,
        description=book_in.description,
        author=book_in.author
    )
    if book_in.category_ids:
        cats = db.query(models.Category).filter(models.Category.id.in_(book_in.category_ids)).all()
        book.categories = cats
    db.add(book)
    db.commit()
    db.refresh(book)
    return book

@router.get("/", response_model=List[schemas.BookOut])
def list_books(category: Optional[int] = None, db: Session = Depends(deps.get_db)):
    query = db.query(models.Book)
    if category:
        query = query.join(models.Book.categories).filter(models.Category.id == category)
    return query.all()

@router.get("/{book_id}", response_model=schemas.BookOut)
def get_book(book_id: int, db: Session = Depends(deps.get_db)):
    book = db.query(models.Book).get(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book
