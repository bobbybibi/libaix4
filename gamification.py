"""
gamification.py — Achievement, XP, leveling, and quiz system for libaix.

Provides:
  • XP earning and level progression
  • Achievement badges (unlock on milestones)
  • Quiz mode with scoring
  • Daily streak tracking
  • Learning path progress
  • Session-based state (stored in Flask session)
"""

from __future__ import annotations

import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date

from knowledge_base import KNOWLEDGE, get_domains

# ── Level thresholds ──────────────────────────────────────────────────

LEVEL_THRESHOLDS: list[int] = [0, 100, 300, 600, 1000, 1500, 2500, 4000, 6000, 10000]

# ── Achievement definitions ───────────────────────────────────────────

ACHIEVEMENTS: dict[str, dict] = {
    "first_question": {"name": "Curious Mind", "desc": "Asked your first question", "icon": "❓", "xp": 50},
    "ten_questions": {"name": "Knowledge Seeker", "desc": "Asked 10 questions", "icon": "🔍", "xp": 100},
    "fifty_questions": {"name": "Scholar", "desc": "Asked 50 questions", "icon": "📚", "xp": 200},
    "first_correct": {"name": "Sharp Eye", "desc": "Got a high-confidence answer", "icon": "🎯", "xp": 50},
    "domain_explorer": {"name": "Domain Explorer", "desc": "Asked about 5 different domains", "icon": "🗺️", "xp": 150},
    "domain_master": {"name": "Domain Master", "desc": "Asked 10+ questions in one domain", "icon": "👑", "xp": 200},
    "streak_3": {"name": "On Fire", "desc": "3-day streak", "icon": "🔥", "xp": 100},
    "streak_7": {"name": "Unstoppable", "desc": "7-day streak", "icon": "⚡", "xp": 250},
    "quiz_perfect": {"name": "Perfect Score", "desc": "Got 100% on a quiz", "icon": "💯", "xp": 300},
    "quiz_five": {"name": "Quiz Champion", "desc": "Completed 5 quizzes", "icon": "🏆", "xp": 200},
    "all_domains": {"name": "Polymath", "desc": "Asked about every domain", "icon": "🌟", "xp": 500},
    "level_5": {"name": "Rising Star", "desc": "Reached level 5", "icon": "⭐", "xp": 0},
    "level_10": {"name": "Grandmaster", "desc": "Reached level 10", "icon": "💎", "xp": 0},
}


# ── Game state ────────────────────────────────────────────────────────

@dataclass
class GameState:
    xp: int = 0
    level: int = 1
    achievements: list[str] = field(default_factory=list)
    questions_asked: int = 0
    correct_domains: dict[str, int] = field(default_factory=dict)
    streak_days: int = 0
    last_active: str = ""
    quiz_scores: list[dict] = field(default_factory=list)
    domains_mastered: list[str] = field(default_factory=list)


# ── Serialization ─────────────────────────────────────────────────────

def load_game_state(session_data: dict) -> GameState:
    """Deserialize a GameState from a Flask session dict."""
    gs = session_data.get("game_state")
    if not gs or not isinstance(gs, dict):
        return GameState()
    return GameState(
        xp=int(gs.get("xp", 0)),
        level=int(gs.get("level", 1)),
        achievements=list(gs.get("achievements", [])),
        questions_asked=int(gs.get("questions_asked", 0)),
        correct_domains=dict(gs.get("correct_domains", {})),
        streak_days=int(gs.get("streak_days", 0)),
        last_active=str(gs.get("last_active", "")),
        quiz_scores=list(gs.get("quiz_scores", [])),
        domains_mastered=list(gs.get("domains_mastered", [])),
    )


def save_game_state(state: GameState) -> dict:
    """Serialize a GameState to a plain dict for Flask session storage."""
    return {
        "xp": state.xp,
        "level": state.level,
        "achievements": state.achievements,
        "questions_asked": state.questions_asked,
        "correct_domains": state.correct_domains,
        "streak_days": state.streak_days,
        "last_active": state.last_active,
        "quiz_scores": state.quiz_scores,
        "domains_mastered": state.domains_mastered,
    }


# ── Core functions ────────────────────────────────────────────────────

def calculate_level(xp: int) -> int:
    """Map XP to a level using LEVEL_THRESHOLDS."""
    level = 1
    for i, threshold in enumerate(LEVEL_THRESHOLDS):
        if xp >= threshold:
            level = i + 1
        else:
            break
    return min(level, len(LEVEL_THRESHOLDS))


def award_xp(state: GameState, amount: int, reason: str) -> tuple[GameState, list[str]]:
    """Add XP, check for level ups and new achievements.

    Returns (updated_state, list_of_new_achievement_keys).
    """
    state.xp += amount
    state.level = calculate_level(state.xp)
    new_achievements = check_achievements(state)
    # Award bonus XP from newly unlocked achievements
    for ach_key in new_achievements:
        bonus = ACHIEVEMENTS[ach_key]["xp"]
        if bonus > 0:
            state.xp += bonus
            state.level = calculate_level(state.xp)
    return state, new_achievements


def check_achievements(state: GameState) -> list[str]:
    """Check and unlock any earned achievements. Returns newly unlocked keys."""
    newly_unlocked: list[str] = []

    def _unlock(key: str) -> None:
        if key not in state.achievements:
            state.achievements.append(key)
            newly_unlocked.append(key)

    # Question milestones
    if state.questions_asked >= 1:
        _unlock("first_question")
    if state.questions_asked >= 10:
        _unlock("ten_questions")
    if state.questions_asked >= 50:
        _unlock("fifty_questions")

    # High-confidence answer
    # (tracked externally — checked via correct_domains having any entry)
    if sum(state.correct_domains.values()) >= 1:
        _unlock("first_correct")

    # Domain exploration
    if len(state.correct_domains) >= 5:
        _unlock("domain_explorer")

    # Domain mastery
    for domain, count in state.correct_domains.items():
        if count >= 10:
            if domain not in state.domains_mastered:
                state.domains_mastered.append(domain)
            _unlock("domain_master")

    # All domains
    all_doms = set(get_domains())
    if all_doms and all_doms <= set(state.correct_domains.keys()):
        _unlock("all_domains")

    # Streak achievements
    if state.streak_days >= 3:
        _unlock("streak_3")
    if state.streak_days >= 7:
        _unlock("streak_7")

    # Quiz achievements
    for qs in state.quiz_scores:
        if qs.get("percent", 0) >= 100.0:
            _unlock("quiz_perfect")
            break
    if len(state.quiz_scores) >= 5:
        _unlock("quiz_five")

    # Level achievements
    if state.level >= 5:
        _unlock("level_5")
    if state.level >= 10:
        _unlock("level_10")

    return newly_unlocked


def update_streak(state: GameState) -> GameState:
    """Check if today continues the streak or breaks it."""
    today = date.today().isoformat()
    if not state.last_active:
        state.streak_days = 1
        state.last_active = today
        return state

    if state.last_active == today:
        return state  # already counted today

    try:
        last = date.fromisoformat(state.last_active)
    except (ValueError, TypeError):
        state.streak_days = 1
        state.last_active = today
        return state

    diff = (date.today() - last).days
    if diff == 1:
        state.streak_days += 1
    elif diff > 1:
        state.streak_days = 1  # streak broken

    state.last_active = today
    return state


def record_question(state: GameState, domain: str, confidence: float) -> tuple[GameState, list[dict]]:
    """Called after each /chat. Returns (state, events[]) where events are XP gains and unlocks."""
    events: list[dict] = []

    state = update_streak(state)
    state.questions_asked += 1

    # Track domain
    if confidence >= 0.15:
        state.correct_domains[domain] = state.correct_domains.get(domain, 0) + 1

    # Base XP for asking
    xp_amount = 10
    if confidence >= 0.6:
        xp_amount = 25  # high-confidence bonus
    elif confidence >= 0.3:
        xp_amount = 15

    state, new_ach = award_xp(state, xp_amount, "question")
    events.append({"type": "xp", "amount": xp_amount, "reason": "question"})

    for ach_key in new_ach:
        ach = ACHIEVEMENTS[ach_key]
        events.append({
            "type": "achievement",
            "key": ach_key,
            "name": ach["name"],
            "desc": ach["desc"],
            "icon": ach["icon"],
            "xp_bonus": ach["xp"],
        })

    return state, events


# ── Quiz system ───────────────────────────────────────────────────────

# In-memory quiz cache keyed by quiz_id
_active_quizzes: dict[str, list[dict]] = {}
_quiz_lock = threading.Lock()


def generate_quiz(domain: str | None = None, count: int = 5) -> list[dict]:
    """Generate quiz questions from the KNOWLEDGE base.

    Each item: {"question": str, "options": [str, ...], "correct": int, "domain": str}
    """
    # Filter entries by domain if specified
    entries = KNOWLEDGE
    if domain:
        entries = [(q, a, d) for q, a, d in KNOWLEDGE if d == domain]

    if len(entries) < count:
        entries = KNOWLEDGE  # fallback to all if not enough in domain

    # Pick random entries
    selected = random.sample(entries, min(count, len(entries)))

    quiz: list[dict] = []
    for question, correct_answer, d in selected:
        # Build wrong options from other answers in the same domain first
        wrong_pool = [a for _, a, dom in KNOWLEDGE if a != correct_answer and dom == d]
        if len(wrong_pool) < 3:
            wrong_pool = [a for _, a, _ in KNOWLEDGE if a != correct_answer]
        wrong = random.sample(wrong_pool, min(3, len(wrong_pool)))
        options = wrong + [correct_answer]
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        quiz.append({
            "question": question,
            "options": options,
            "correct": correct_idx,
            "domain": d,
        })

    return quiz


def score_quiz(answers: list[int], quiz: list[dict]) -> dict:
    """Score a quiz attempt.

    Returns {"score": int, "total": int, "percent": float, "xp_earned": int}
    """
    total = len(quiz)
    if total == 0:
        return {"score": 0, "total": 0, "percent": 0.0, "xp_earned": 0}

    score = 0
    for i, q in enumerate(quiz):
        if i < len(answers) and answers[i] == q["correct"]:
            score += 1

    percent = round((score / total) * 100, 1)
    # XP: 20 per correct answer, bonus for perfect
    xp_earned = score * 20
    if score == total:
        xp_earned += 50  # perfect score bonus

    return {
        "score": score,
        "total": total,
        "percent": percent,
        "xp_earned": xp_earned,
    }


def get_leaderboard_entry(state: GameState) -> dict:
    """Get formatted stats for display."""
    next_level_xp = LEVEL_THRESHOLDS[state.level] if state.level < len(LEVEL_THRESHOLDS) else LEVEL_THRESHOLDS[-1]
    prev_level_xp = LEVEL_THRESHOLDS[state.level - 1] if state.level > 0 else 0
    xp_in_level = state.xp - prev_level_xp
    xp_needed = next_level_xp - prev_level_xp if next_level_xp > prev_level_xp else 1
    progress = min(round((xp_in_level / xp_needed) * 100, 1), 100.0) if xp_needed > 0 else 100.0

    return {
        "xp": state.xp,
        "level": state.level,
        "level_progress": progress,
        "next_level_xp": next_level_xp,
        "achievements": [
            {
                "key": k,
                "name": ACHIEVEMENTS[k]["name"],
                "icon": ACHIEVEMENTS[k]["icon"],
                "desc": ACHIEVEMENTS[k]["desc"],
            }
            for k in state.achievements
            if k in ACHIEVEMENTS
        ],
        "total_achievements": len(state.achievements),
        "questions_asked": state.questions_asked,
        "streak_days": state.streak_days,
        "domains_explored": len(state.correct_domains),
        "domains_mastered": state.domains_mastered,
        "quizzes_taken": len(state.quiz_scores),
    }


def store_quiz(quiz: list[dict]) -> str:
    """Store a quiz in memory and return its ID."""
    quiz_id = uuid.uuid4().hex[:12]
    with _quiz_lock:
        _active_quizzes[quiz_id] = quiz
        # Prune old quizzes if cache grows too large
        if len(_active_quizzes) > 1000:
            oldest = list(_active_quizzes.keys())[:500]
            for k in oldest:
                _active_quizzes.pop(k, None)
    return quiz_id


def get_stored_quiz(quiz_id: str) -> list[dict] | None:
    """Retrieve a stored quiz by ID."""
    with _quiz_lock:
        return _active_quizzes.get(quiz_id)
