from flask import Flask
from sqlalchemy import inspect, text

from config import DATABASE_URL, SQLALCHEMY_TRACK_MODIFICATIONS
from database import db
import models  # noqa: F401


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
db.init_app(app)


def ensure_database_schema() -> None:
    inspector = inspect(db.engine)
    if not inspector.has_table("users"):
        return

    user_columns = {column["name"] for column in inspector.get_columns("users")}
    if "student_grade" not in user_columns:
        with db.engine.begin() as connection:
            connection.execute(text("ALTER TABLE users ADD COLUMN student_grade INTEGER"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        ensure_database_schema()
        print("Database tables created successfully.")
        print(f"Database URI: {DATABASE_URL}")
