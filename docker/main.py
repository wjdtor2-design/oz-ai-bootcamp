from fastapi import FastAPI
from sqlalchemy import text

from database import SessionFactory

app = FastAPI()

@app.get("/users")
async def get_users_handler():
    with SessionFactory() as session:
        stmt = text("SELECT * FROM user;")
        result = session.execute(stmt).mappings().all()
    return {"result": result}