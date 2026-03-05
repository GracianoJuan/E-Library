from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas, deps, auth

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/me", response_model=schemas.UserOut)
def read_current_user(current_user: models.User = Depends(deps.get_current_user)):
    return current_user

# additional user-related endpoints can be added here
 