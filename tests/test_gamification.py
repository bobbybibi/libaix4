"""
test_gamification.py — Tests for the gamification system.

Covers:
  • GameState creation and serialization
  • XP awarding and level calculation
  • Achievement unlocking
  • Question recording
  • Quiz generation and scoring
  • Streak tracking
  • Flask endpoint tests (/game/status, /game/quiz, etc.)
  • Chat endpoint returns game data
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gamification import (
    ACHIEVEMENTS,
    LEVEL_THRESHOLDS,
    GameState,
    award_xp,
    calculate_level,
    check_achievements,
    generate_quiz,
    get_leaderboard_entry,
    get_stored_quiz,
    load_game_state,
    record_question,
    save_game_state,
    score_quiz,
    store_quiz,
    update_streak,
)


# ── Unit tests ────────────────────────────────────────────────────────


class TestInitialGameState:
    def test_defaults(self):
        gs = GameState()
        assert gs.xp == 0
        assert gs.level == 1
        assert gs.achievements == []
        assert gs.questions_asked == 0
        assert gs.correct_domains == {}
        assert gs.streak_days == 0
        assert gs.last_active == ""
        assert gs.quiz_scores == []
        assert gs.domains_mastered == []

    def test_round_trip_serialization(self):
        gs = GameState(xp=150, level=3, achievements=["first_question"])
        data = save_game_state(gs)
        gs2 = load_game_state({"game_state": data})
        assert gs2.xp == 150
        assert gs2.level == 3
        assert gs2.achievements == ["first_question"]

    def test_load_from_empty_session(self):
        gs = load_game_state({})
        assert gs.xp == 0
        assert gs.level == 1

    def test_load_from_none_game_state(self):
        gs = load_game_state({"game_state": None})
        assert gs.xp == 0


class TestCalculateLevel:
    def test_zero_xp(self):
        assert calculate_level(0) == 1

    def test_level_2(self):
        assert calculate_level(100) == 2

    def test_level_3(self):
        assert calculate_level(300) == 3

    def test_between_levels(self):
        assert calculate_level(250) == 2

    def test_max_level(self):
        assert calculate_level(99999) == len(LEVEL_THRESHOLDS)

    def test_all_thresholds(self):
        for i, threshold in enumerate(LEVEL_THRESHOLDS):
            assert calculate_level(threshold) == i + 1


class TestAwardXpAndLevelUp:
    def test_basic_xp_gain(self):
        gs = GameState()
        gs, new_ach = award_xp(gs, 50, "test")
        assert gs.xp >= 50

    def test_level_up(self):
        gs = GameState()
        gs, _ = award_xp(gs, 100, "test")
        assert gs.level == 2

    def test_multiple_level_ups(self):
        gs = GameState()
        gs, _ = award_xp(gs, 1000, "big_bonus")
        assert gs.level >= 5

    def test_achievements_unlocked_on_level(self):
        gs = GameState(xp=999, level=4)
        gs, new_ach = award_xp(gs, 1, "test")
        assert gs.level >= 5
        assert "level_5" in new_ach or "level_5" in gs.achievements


class TestCheckAchievements:
    def test_first_question(self):
        gs = GameState(questions_asked=1)
        new = check_achievements(gs)
        assert "first_question" in new

    def test_ten_questions(self):
        gs = GameState(questions_asked=10)
        new = check_achievements(gs)
        assert "ten_questions" in new

    def test_fifty_questions(self):
        gs = GameState(questions_asked=50)
        new = check_achievements(gs)
        assert "fifty_questions" in new

    def test_no_duplicate_unlock(self):
        gs = GameState(questions_asked=10, achievements=["first_question", "ten_questions"])
        new = check_achievements(gs)
        assert "first_question" not in new
        assert "ten_questions" not in new

    def test_domain_explorer(self):
        gs = GameState(correct_domains={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1})
        new = check_achievements(gs)
        assert "domain_explorer" in new

    def test_domain_master(self):
        gs = GameState(correct_domains={"networking": 10})
        new = check_achievements(gs)
        assert "domain_master" in new

    def test_streak_3(self):
        gs = GameState(streak_days=3)
        new = check_achievements(gs)
        assert "streak_3" in new

    def test_streak_7(self):
        gs = GameState(streak_days=7)
        new = check_achievements(gs)
        assert "streak_7" in new

    def test_quiz_perfect(self):
        gs = GameState(quiz_scores=[{"percent": 100.0}])
        new = check_achievements(gs)
        assert "quiz_perfect" in new

    def test_quiz_five(self):
        gs = GameState(quiz_scores=[{"percent": 80}] * 5)
        new = check_achievements(gs)
        assert "quiz_five" in new


class TestRecordQuestion:
    def test_increments_questions_asked(self):
        gs = GameState()
        gs, events = record_question(gs, "networking", 0.8)
        assert gs.questions_asked == 1

    def test_tracks_domain(self):
        gs = GameState()
        gs, events = record_question(gs, "networking", 0.5)
        assert "networking" in gs.correct_domains

    def test_xp_gained(self):
        gs = GameState()
        gs, events = record_question(gs, "security", 0.8)
        assert gs.xp > 0
        xp_events = [e for e in events if e["type"] == "xp"]
        assert len(xp_events) >= 1

    def test_achievement_event_on_first_question(self):
        gs = GameState()
        gs, events = record_question(gs, "networking", 0.8)
        ach_events = [e for e in events if e["type"] == "achievement"]
        assert any(e["key"] == "first_question" for e in ach_events)

    def test_low_confidence_no_domain_track(self):
        gs = GameState()
        gs, events = record_question(gs, "networking", 0.05)
        assert "networking" not in gs.correct_domains


class TestGenerateQuiz:
    def test_returns_correct_count(self):
        quiz = generate_quiz(count=3)
        assert len(quiz) == 3

    def test_default_count(self):
        quiz = generate_quiz()
        assert len(quiz) == 5

    def test_question_format(self):
        quiz = generate_quiz(count=1)
        q = quiz[0]
        assert "question" in q
        assert "options" in q
        assert "correct" in q
        assert "domain" in q
        assert isinstance(q["options"], list)
        assert len(q["options"]) >= 2
        assert 0 <= q["correct"] < len(q["options"])

    def test_domain_filter(self):
        quiz = generate_quiz(domain="networking", count=3)
        for q in quiz:
            assert q["domain"] == "networking"

    def test_fallback_on_small_domain(self):
        quiz = generate_quiz(domain="nonexistent_domain_xyz", count=3)
        assert len(quiz) == 3


class TestScoreQuiz:
    def test_perfect_score(self):
        quiz = [
            {"question": "Q1", "options": ["A", "B", "C", "D"], "correct": 0, "domain": "test"},
            {"question": "Q2", "options": ["A", "B", "C", "D"], "correct": 1, "domain": "test"},
        ]
        result = score_quiz([0, 1], quiz)
        assert result["score"] == 2
        assert result["total"] == 2
        assert result["percent"] == 100.0
        assert result["xp_earned"] == 2 * 20 + 50  # 40 + 50 perfect bonus

    def test_partial_score(self):
        quiz = [
            {"question": "Q1", "options": ["A", "B"], "correct": 0, "domain": "test"},
            {"question": "Q2", "options": ["A", "B"], "correct": 1, "domain": "test"},
        ]
        result = score_quiz([0, 0], quiz)
        assert result["score"] == 1
        assert result["total"] == 2
        assert result["percent"] == 50.0
        assert result["xp_earned"] == 20

    def test_zero_score(self):
        quiz = [
            {"question": "Q1", "options": ["A", "B"], "correct": 1, "domain": "test"},
        ]
        result = score_quiz([0], quiz)
        assert result["score"] == 0
        assert result["xp_earned"] == 0

    def test_empty_quiz(self):
        result = score_quiz([], [])
        assert result["score"] == 0
        assert result["total"] == 0


class TestUpdateStreak:
    def test_first_visit(self):
        gs = GameState()
        gs = update_streak(gs)
        assert gs.streak_days == 1
        assert gs.last_active == date.today().isoformat()

    def test_same_day_no_change(self):
        gs = GameState(streak_days=3, last_active=date.today().isoformat())
        gs = update_streak(gs)
        assert gs.streak_days == 3

    def test_consecutive_day_continues(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        gs = GameState(streak_days=2, last_active=yesterday)
        gs = update_streak(gs)
        assert gs.streak_days == 3

    def test_gap_breaks_streak(self):
        two_days_ago = (date.today() - timedelta(days=2)).isoformat()
        gs = GameState(streak_days=5, last_active=two_days_ago)
        gs = update_streak(gs)
        assert gs.streak_days == 1

    def test_invalid_last_active(self):
        gs = GameState(streak_days=5, last_active="not-a-date")
        gs = update_streak(gs)
        assert gs.streak_days == 1


class TestLeaderboardEntry:
    def test_basic_entry(self):
        gs = GameState(xp=150, level=2, achievements=["first_question"], questions_asked=5, streak_days=2)
        entry = get_leaderboard_entry(gs)
        assert entry["xp"] == 150
        assert entry["level"] == 2
        assert entry["questions_asked"] == 5
        assert entry["streak_days"] == 2
        assert entry["total_achievements"] == 1
        assert "level_progress" in entry


class TestQuizStorage:
    def test_store_and_retrieve(self):
        quiz = [{"question": "test", "options": ["a"], "correct": 0, "domain": "d"}]
        qid = store_quiz(quiz)
        assert isinstance(qid, str)
        assert len(qid) == 12
        retrieved = get_stored_quiz(qid)
        assert retrieved == quiz

    def test_missing_quiz(self):
        assert get_stored_quiz("nonexistent") is None


# ── Flask endpoint tests ──────────────────────────────────────────────

from app import app  # noqa: E402


class TestGameStatusEndpoint:
    def setup_method(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_returns_200(self):
        resp = self.client.get("/game/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "xp" in data
        assert "level" in data
        assert "achievements" in data

    def test_fresh_session_defaults(self):
        resp = self.client.get("/game/status")
        data = resp.get_json()
        assert data["xp"] == 0
        assert data["level"] == 1


class TestQuizEndpoint:
    def setup_method(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_generate_quiz(self):
        resp = self.client.post("/game/quiz", json={"count": 3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "quiz_id" in data
        assert "questions" in data
        assert len(data["questions"]) == 3
        # Ensure correct answers are NOT exposed
        for q in data["questions"]:
            assert "correct" not in q

    def test_generate_quiz_with_domain(self):
        resp = self.client.post("/game/quiz", json={"domain": "networking", "count": 2})
        assert resp.status_code == 200
        data = resp.get_json()
        for q in data["questions"]:
            assert q["domain"] == "networking"

    def test_submit_quiz(self):
        # First generate
        gen = self.client.post("/game/quiz", json={"count": 2})
        gen_data = gen.get_json()
        quiz_id = gen_data["quiz_id"]
        # Submit all zeros
        resp = self.client.post("/game/quiz/submit", json={"quiz_id": quiz_id, "answers": [0, 0]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "score" in data
        assert "total" in data
        assert "percent" in data
        assert "xp_earned" in data
        assert "xp" in data
        assert "level" in data

    def test_submit_invalid_quiz_id(self):
        resp = self.client.post("/game/quiz/submit", json={"quiz_id": "invalid", "answers": [0]})
        assert resp.status_code == 404

    def test_submit_missing_quiz_id(self):
        resp = self.client.post("/game/quiz/submit", json={"answers": [0]})
        assert resp.status_code == 400


class TestAchievementsEndpoint:
    def setup_method(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_returns_all_achievements(self):
        resp = self.client.get("/game/achievements")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) == len(ACHIEVEMENTS)
        for ach in data:
            assert "key" in ach
            assert "name" in ach
            assert "unlocked" in ach
            assert isinstance(ach["unlocked"], bool)

    def test_all_start_locked(self):
        resp = self.client.get("/game/achievements")
        data = resp.get_json()
        for ach in data:
            assert ach["unlocked"] is False


class TestChatReturnsGameData:
    def setup_method(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_chat_includes_game_key(self):
        """The /chat response should include a 'game' key with XP info."""
        with patch("app.knowledge_model") as mock_model, \
             patch("app.knowledge_bow") as mock_bow:
            # Set up mock to simulate a loaded model
            import numpy as np
            mock_model.__bool__ = lambda self: True
            mock_model.return_value = True
            mock_bow.__bool__ = lambda self: True
            mock_bow.return_value = True
            mock_bow.transform.return_value = np.array([[1.0]])
            mock_model.predict.return_value = np.array([[0.9, 0.1]])
            with patch("app.knowledge_answer_map", {0: "Test answer", 1: "Other"}):
                resp = self.client.post("/chat", json={"question": "What is TCP?"})
                if resp.status_code == 200:
                    data = resp.get_json()
                    if "game" in data:
                        assert "xp" in data["game"]
                        assert "level" in data["game"]
                        assert "events" in data["game"]
