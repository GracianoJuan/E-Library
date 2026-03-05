from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

class UserController:
    def create_user():
        print("Creating user...")