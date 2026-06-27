"""api_v1.py — versioned SaaS API for libaix web/mobile clients."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request, session

from saas_db import (
    AnalyticsEvent,
    Subscription,
    Tenant,
    User,
    current_tenant,
    current_user,
    db,
    ensure_unique_slug,
    slugify_name,
)

api_v1_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")



def _is_valid_email(email: str) -> bool:
    if not email or " " in email or "\t" in email or "\n" in email:
        return False
    if email.count("@") != 1:
        return False
    local, domain = email.split("@", 1)
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    if domain.startswith(".") or domain.endswith("."):
        return False
    return True


_PLAN_DEFS = {
    "free": {"name": "Free", "monthly_usd": 0, "max_requests_per_day": 150},
    "pro": {"name": "Pro", "monthly_usd": 19, "max_requests_per_day": 3000},
    "team": {"name": "Team", "monthly_usd": 79, "max_requests_per_day": 20000},
}


def _require_user():
    user = current_user()
    if user is None or not user.is_active:
        return None, (jsonify({"error": "Authentication required"}), 401)
    return user, None


def _trade_pack_summary(trade_id: str) -> dict | None:
    try:
        import trade_pack

        pack = trade_pack.load_trade(trade_id)
        if not isinstance(pack, dict):
            return None
        return {
            "id": trade_id,
            "name": str(pack.get("name", trade_id)),
            "domains": list(pack.get("domains") or []),
            "disclaimers": list(pack.get("disclaimers") or []),
        }
    except Exception:
        return None


def _proxy_json_post(path: str, payload: dict) -> tuple[dict, int]:
    with current_app.test_client() as tc:
        r = tc.post(path, json=payload)
    try:
        data = r.get_json(silent=True)
    except Exception:
        data = None
    if not isinstance(data, dict):
        data = {"error": "Unexpected upstream response"}
    return data, r.status_code


@api_v1_bp.route("/health", methods=["GET"])
def health_v1():
    db_ok = True
    try:
        _ = db.session.query(Tenant.id).limit(1).all()
    except Exception:
        db_ok = False
    return jsonify({"status": "ok" if db_ok else "degraded", "db": db_ok, "version": "v1"})


@api_v1_bp.route("/trade-packs", methods=["GET"])
def trade_packs():
    try:
        import trade_pack

        trades = []
        for tid in trade_pack.list_trades():
            summary = _trade_pack_summary(tid)
            if summary:
                trades.append(summary)

        active_tid = trade_pack.get_active_trade_id()
        return jsonify({"active_trade": active_tid, "trades": trades})
    except Exception:
        return jsonify({"active_trade": "networking", "trades": []})


@api_v1_bp.route("/config", methods=["GET"])
def config_v1():
    tenant = current_tenant()
    user = current_user()
    active_trade = tenant.trade_id if tenant else "networking"
    summary = _trade_pack_summary(active_trade)
    return jsonify(
        {
            "active_trade": active_trade,
            "trade": summary,
            "tenant": {"id": tenant.id, "name": tenant.name, "plan": tenant.plan} if tenant else None,
            "user": {"id": user.id, "email": user.email, "role": user.role} if user else None,
        }
    )


@api_v1_bp.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    tenant_name = str(data.get("tenant_name", "")).strip() or email.split("@")[0]
    trade_id = str(data.get("trade_id", "networking")).strip() or "networking"

    if not _is_valid_email(email):
        return jsonify({"error": "Valid email is required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if User.query.filter_by(email=email).first() is not None:
        return jsonify({"error": "Email already registered"}), 409

    base_slug = slugify_name(tenant_name)
    tenant = Tenant(name=tenant_name, slug=ensure_unique_slug(base_slug), trade_id=trade_id, plan="free")
    db.session.add(tenant)
    db.session.flush()

    user = User(tenant_id=tenant.id, email=email, role="owner", is_active=True)
    user.set_password(password)
    db.session.add(user)

    db.session.add(Subscription(tenant_id=tenant.id, plan="free", status="active", provider="none"))
    db.session.commit()

    session["user_id"] = user.id
    session["tenant_id"] = tenant.id
    return jsonify({"status": "ok", "tenant_id": tenant.id, "user_id": user.id, "plan": "free"}), 201


@api_v1_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password) or not user.is_active:
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = user.id
    session["tenant_id"] = user.tenant_id
    return jsonify({"status": "ok", "user_id": user.id, "tenant_id": user.tenant_id})


@api_v1_bp.route("/auth/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    session.pop("tenant_id", None)
    return jsonify({"status": "ok"})


@api_v1_bp.route("/me", methods=["GET"])
def me():
    user, err = _require_user()
    if err:
        return err
    tenant = current_tenant()
    return jsonify(
        {
            "user": {"id": user.id, "email": user.email, "role": user.role},
            "tenant": {
                "id": tenant.id,
                "name": tenant.name,
                "slug": tenant.slug,
                "plan": tenant.plan,
                "trade_id": tenant.trade_id,
            }
            if tenant
            else None,
            "plans": _PLAN_DEFS,
        }
    )


@api_v1_bp.route("/onboarding/select-trade", methods=["POST"])
def onboarding_select_trade():
    user, err = _require_user()
    if err:
        return err

    data = request.get_json(force=True)
    trade_id = str(data.get("trade_id", "")).strip()
    summary = _trade_pack_summary(trade_id)
    if summary is None:
        return jsonify({"error": "Unknown trade_id"}), 400

    tenant = Tenant.query.get(user.tenant_id)
    if tenant is None:
        return jsonify({"error": "Tenant not found"}), 404
    tenant.trade_id = trade_id
    db.session.commit()

    return jsonify({"status": "ok", "tenant_id": tenant.id, "trade": summary})


@api_v1_bp.route("/billing/plans", methods=["GET"])
def billing_plans():
    return jsonify({"plans": _PLAN_DEFS})


@api_v1_bp.route("/billing/subscribe", methods=["POST"])
def billing_subscribe():
    user, err = _require_user()
    if err:
        return err

    data = request.get_json(force=True)
    plan = str(data.get("plan", "")).strip().lower()
    if plan not in _PLAN_DEFS:
        return jsonify({"error": "Unknown plan"}), 400

    tenant = Tenant.query.get(user.tenant_id)
    if tenant is None:
        return jsonify({"error": "Tenant not found"}), 404

    sub = Subscription.query.filter_by(tenant_id=tenant.id).first()
    if sub is None:
        sub = Subscription(tenant_id=tenant.id, plan=plan, status="active", provider="stripe")
        db.session.add(sub)
    else:
        sub.plan = plan
        sub.status = "active"
        sub.provider = "stripe"
        sub.updated_at = datetime.now(timezone.utc)
    tenant.plan = plan
    db.session.commit()

    return jsonify({"status": "ok", "plan": plan, "tenant_id": tenant.id})


@api_v1_bp.route("/chat", methods=["POST"])
def chat_v1():
    user, err = _require_user()
    if err:
        return err

    data = request.get_json(force=True)
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    payload, status = _proxy_json_post("/chat", {"question": question})
    payload["tenant_id"] = user.tenant_id
    return jsonify(payload), status


@api_v1_bp.route("/research", methods=["POST"])
def research_v1():
    user, err = _require_user()
    if err:
        return err

    data = request.get_json(force=True)
    topic = str(data.get("topic", "")).strip()
    urls = data.get("urls", [])
    payload, status = _proxy_json_post("/chat/research", {"topic": topic, "urls": urls})
    payload["tenant_id"] = user.tenant_id
    return jsonify(payload), status


@api_v1_bp.route("/analytics/query-feedback", methods=["POST"])
def analytics_query_feedback():
    data = request.get_json(force=True)
    tenant = current_tenant()
    user = current_user()

    confidence = float(data.get("confidence", 0.0))
    answered = bool(data.get("answered", True))
    outcome = "good"
    if not answered:
        outcome = "unanswered"
    elif confidence < 0.15:
        outcome = "low_confidence"

    payload = {
        "question": str(data.get("question", ""))[:2000],
        "domain": str(data.get("domain", "general"))[:120],
        "confidence": confidence,
        "answered": answered,
        "outcome": outcome,
        "strategy": str(data.get("strategy", ""))[:120],
    }

    event = AnalyticsEvent(
        tenant_id=tenant.id if tenant else None,
        user_id=user.id if user else None,
        event_type="query_feedback",
        payload_json=json.dumps(payload, ensure_ascii=False),
    )
    db.session.add(event)
    db.session.commit()

    return jsonify({"status": "ok", "outcome": outcome})
