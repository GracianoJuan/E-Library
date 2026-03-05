from fastapi import FastAPI
# from module.routes.UserRoutes import router as UserRouter
# from module.routes import auth, books, categories, likes

# from database import engine
# from module.models import Base  # noqa: F401 ensure models are imported so metadata is available

app = FastAPI()

# # create tables automatically (for development; production should use migrations)
# Base.metadata.create_all(bind=engine)

# # include routers
# app.include_router(UserRouter)
# app.include_router(auth.router)
# app.include_router(books.router)
# app.include_router(categories.router)
# app.include_router(likes.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}