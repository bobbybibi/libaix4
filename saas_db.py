"""saas_db.py — multi-tenant SaaS data layer for libaix."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from flask import session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash


db = SQLAlchemy()


class Tenant(db.Model):
    __tablename__ = "tenants"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    slug = db.Column(db.String(120), nullable=False, unique=True, index=True)
    trade_id = db.Column(db.String(80), nullable=False, default="networking")
    plan = db.Column(db.String(32), nullable=False, default="free")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    users = db.relationship("User", backref="tenant", lazy=True, cascade="all, delete-orphan")


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False, index=True)
    email = db.Column(db.String(255), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(32), nullable=False, default="owner")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)


class Subscription(db.Model):
    __tablename__ = "subscriptions"

    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False, index=True, unique=True)
    plan = db.Column(db.String(32), nullable=False, default="free")
    status = db.Column(db.String(32), nullable=False, default="active")
    provider = db.Column(db.String(32), nullable=False, default="none")
    external_id = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class AnalyticsEvent(db.Model):
    __tablename__ = "analytics_events"

    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    event_type = db.Column(db.String(80), nullable=False, index=True)
    payload_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


def slugify_name(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:120] or "tenant"


def ensure_unique_slug(base_slug: str) -> str:
    slug = base_slug
    n = 2
    while Tenant.query.filter_by(slug=slug).first() is not None:
        slug = f"{base_slug}-{n}"
        n += 1
    return slug


def current_user() -> User | None:
    uid = session.get("user_id")
    if not isinstance(uid, int):
        return None
    return User.query.get(uid)


def current_tenant() -> Tenant | None:
    tid = session.get("tenant_id")
    if not isinstance(tid, int):
        return None
    return Tenant.query.get(tid)


def init_saas_app(app) -> None:
    default_db_path = (Path(__file__).resolve().parent / "data" / "libaix_saas.db").as_posix()
    app.config.setdefault("SQLALCHEMY_DATABASE_URI", f"sqlite:///{default_db_path}")
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)

    uri = str(app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    if uri.startswith("sqlite:///"):
        sqlite_path = Path(uri.replace("sqlite:///", "", 1))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()
