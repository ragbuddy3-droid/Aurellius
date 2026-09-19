from flask import Flask
from werkzeug.security import generate_password_hash

from config import DATABASE_URL, SQLALCHEMY_TRACK_MODIFICATIONS
from database import db
from models import User


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
db.init_app(app)


DEFAULT_USERS = [
    {
        "name": "Guru Admin",
        "username": "guruadmin",
        "password": "admin123",
        "role": "admin",
        "student_grade": None,
    },
    {
        "name": "Siswa Demo",
        "username": "siswa1",
        "password": "user123",
        "role": "user",
        "student_grade": 4,
    },
]


if __name__ == "__main__":
    with app.app_context():
        for item in DEFAULT_USERS:
            existing = User.query.filter_by(username=item["username"]).first()
            if existing:
                if item["student_grade"] is not None and existing.student_grade is None:
                    existing.student_grade = item["student_grade"]
                    db.session.add(existing)
                    print(f"User {item['username']} updated with student grade {item['student_grade']}.")
                else:
                    print(f"User {item['username']} already exists, skipped.")
                continue

            user = User(
                name=item["name"],
                username=item["username"],
                password_hash=generate_password_hash(item["password"]),
                role=item["role"],
                is_active=True,
                student_grade=item["student_grade"],
            )
            db.session.add(user)

        db.session.commit()
        print("Seed users completed.")
        print("Admin: guruadmin / admin123")
        print("User : siswa1 / user123")
