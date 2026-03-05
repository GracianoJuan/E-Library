from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas, deps

router = APIRouter(
    prefix="/likes",
    tags=["likes"]
)

@router.post("/", response_model=schemas.LikeOut)
def like_book(like_in: schemas.LikeBase,
              db: Session = Depends(deps.get_db),
              current_user: models.User = Depends(deps.get_current_user)):
    # ensure the authenticated user matches the like payload
    if current_user.id != like_in.user_id:
        raise HTTPException(status_code=403, detail="Cannot like as another user")
    user = db.query(models.User).get(like_in.user_id)
    book = db.query(models.Book).get(like_in.book_id)
    if not user or not book:
        raise HTTPException(status_code=404, detail="User or book not found")
    existing = db.query(models.Like).filter_by(user_id=like_in.user_id, book_id=like_in.book_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already liked")
    like = models.Like(user_id=like_in.user_id, book_id=like_in.book_id)
    book.total_likes = book.total_likes + 1
    db.add(like)
    db.commit()
    db.refresh(like)
    return like

@router.get("/", response_model=list[schemas.LikeOut])
def list_likes(db: Session = Depends(deps.get_db)):
    return db.query(models.Like).all()
