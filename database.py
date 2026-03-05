import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# read database url from environment variable or use a default for development
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/elib")

# create synchronous engine
engine = create_engine(DATABASE_URL)

# each path operation will get its own Session instance
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# base class for models
Base = declarative_base()
