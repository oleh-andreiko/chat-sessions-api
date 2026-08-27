"""The parts that decide what the model sees and what counts as history.

These are the behaviours that are easy to break by accident and impossible to
notice from the outside: nothing crashes when the wrong messages are replayed,
the answers just quietly stop making sense.
"""

from decimal import Decimal

from app import config
from app.models import Message


def _texts(call):
    return [m["content"] for m in call["messages"]]


def test_the_previous_exchange_is_replayed_to_the_model(
    client, session_id, fake_openai
):
    client.post(f"/sessions/{session_id}/messages", json={"content": "first"})
    client.post(f"/sessions/{session_id}/messages", json={"content": "second"})

    # The second call carries the whole conversation, not just the new line.
    assert _texts(fake_openai[1]) == ["first", "reply 1", "second"]


def test_roles_alternate_in_the_replayed_context(client, session_id, fake_openai):
    client.post(f"/sessions/{session_id}/messages", json={"content": "first"})
    client.post(f"/sessions/{session_id}/messages", json={"content": "second"})

    assert [m["role"] for m in fake_openai[1]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


def test_context_is_capped_at_max_history_messages(client, session_id, fake_openai):
    cap = config.MAX_HISTORY_MESSAGES
    exchanges = cap  # each one adds two rows, so this is twice the cap

    for i in range(exchanges):
        client.post(f"/sessions/{session_id}/messages", json={"content": f"m{i}"})

    # cap past messages plus the new one.
    last_call = fake_openai[-1]
    assert len(last_call["messages"]) == cap + 1
    # And it is the newest ones that survived, not the oldest.
    assert last_call["messages"][-1]["content"] == f"m{exchanges - 1}"
    assert "m0" not in _texts(last_call)


def test_the_whole_history_is_still_stored_when_the_context_is_capped(
    client, session_id, fake_openai
):
    for i in range(config.MAX_HISTORY_MESSAGES):
        client.post(f"/sessions/{session_id}/messages", json={"content": f"m{i}"})

    stored = client.get(f"/sessions/{session_id}").json()["messages"]

    # Trimming decides what the model sees, not what the database keeps.
    assert len(stored) == config.MAX_HISTORY_MESSAGES * 2
    assert stored[0]["content"] == "m0"


def test_the_pair_of_an_exchange_is_stored_with_adjacent_ids(
    client, session_id, fake_openai, db
):
    client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})

    ids = [m.id for m in db.query(Message).order_by(Message.id).all()]
    # Both rows go in one INSERT. If they are ever split into two statements,
    # a concurrent request can land between them and the roles fall out of order.
    assert ids[1] == ids[0] + 1


def test_reset_keeps_the_session_id(client, session_id, fake_openai):
    client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})

    body = client.post(f"/sessions/{session_id}/reset").json()

    assert body["id"] == session_id


def test_reset_empties_the_active_history_and_the_totals(
    client, session_id, fake_openai
):
    client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})

    body = client.post(f"/sessions/{session_id}/reset").json()

    assert body["messages"] == []
    assert body["total_prompt_tokens"] == 0
    assert body["total_completion_tokens"] == 0
    assert Decimal(body["total_cost"]) == Decimal(0)
    assert body["current_generation"] == 2


def test_the_model_does_not_see_anything_from_before_the_reset(
    client, session_id, fake_openai
):
    client.post(f"/sessions/{session_id}/messages", json={"content": "before"})
    client.post(f"/sessions/{session_id}/reset")
    client.post(f"/sessions/{session_id}/messages", json={"content": "after"})

    assert _texts(fake_openai[-1]) == ["after"]


def test_reset_archives_rather_than_deletes(client, session_id, fake_openai, db):
    client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})
    client.post(f"/sessions/{session_id}/reset")

    rows = db.query(Message).all()
    # The record of what was already paid for outlives the reset; it is only
    # excluded from the active context.
    assert len(rows) == 2
    assert {r.generation for r in rows} == {1}


def test_spend_before_the_reset_is_still_recoverable(client, session_id, fake_openai, db):
    client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})
    client.post(f"/sessions/{session_id}/reset")
    client.post(f"/sessions/{session_id}/messages", json={"content": "hi again"})

    active = Decimal(client.get(f"/sessions/{session_id}").json()["total_cost"])
    lifetime = sum((m.cost or Decimal(0) for m in db.query(Message).all()), Decimal(0))

    assert active == Decimal("0.000021")
    assert lifetime == Decimal("0.000042")


def test_resetting_twice_keeps_the_generations_apart(client, session_id, fake_openai, db):
    client.post(f"/sessions/{session_id}/messages", json={"content": "one"})
    client.post(f"/sessions/{session_id}/reset")
    client.post(f"/sessions/{session_id}/messages", json={"content": "two"})
    client.post(f"/sessions/{session_id}/reset")

    generations = sorted(m.generation for m in db.query(Message).all())

    assert generations == [1, 1, 2, 2]
    assert client.get(f"/sessions/{session_id}").json()["current_generation"] == 3


def test_history_endpoint_shows_only_the_active_generation(
    client, session_id, fake_openai
):
    client.post(f"/sessions/{session_id}/messages", json={"content": "old"})
    client.post(f"/sessions/{session_id}/reset")
    client.post(f"/sessions/{session_id}/messages", json={"content": "new"})

    contents = [m["content"] for m in client.get(f"/sessions/{session_id}/messages").json()]

    assert contents == ["new", "reply 2"]


def test_sessions_do_not_see_each_others_messages(client, fake_openai):
    a = client.post("/sessions", json={}).json()["id"]
    b = client.post("/sessions", json={}).json()["id"]

    client.post(f"/sessions/{a}/messages", json={"content": "belongs to a"})
    client.post(f"/sessions/{b}/messages", json={"content": "belongs to b"})

    assert _texts(fake_openai[1]) == ["belongs to b"]
    assert client.get(f"/sessions/{b}").json()["messages"][0]["content"] == "belongs to b"
