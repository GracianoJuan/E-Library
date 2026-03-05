from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas, deps

router = APIRouter(
    prefix="/categories",
    tags=["categories"]
)

@router.post("/", response_model=schemas.CategoryOut)
def create_category(cat_in: schemas.CategoryCreate,
                    db: Session = Depends(deps.get_db),
                    current_user: models.User = Depends(deps.get_current_user)):
    existing = db.query(models.Category).filter(models.Category.name == cat_in.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Category already exists")
    cat = models.Category(name=cat_in.name)
    db.add(cat)
    db.commit()
    db.refresh(cat)
    return cat

@router.get("/", response_model=List[schemas.CategoryOut])
def list_categories(db: Session = Depends(deps.get_db)):
    return db.query(models.Category).all()
