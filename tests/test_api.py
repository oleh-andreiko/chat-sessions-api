"""Endpoint behaviour: status codes, response shape, error bodies."""

from decimal import Decimal


def test_create_session_uses_the_default_model(client):
    response = client.post("/sessions", json={"title": "hello"})

    assert response.status_code == 201
    body = response.json()
    assert body["model"] == "gpt-4o-mini"
    assert body["title"] == "hello"


def test_create_session_accepts_a_supported_model(client):
    response = client.post("/sessions", json={"model": "gpt-4.1-nano"})

    assert response.status_code == 201
    assert response.json()["model"] == "gpt-4.1-nano"


def test_create_session_rejects_an_unsupported_model(client):
    # Caught here rather than on the first message: a session nobody can send
    # to is not worth creating.
    response = client.post("/sessions", json={"model": "gpt-nope"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model"


def test_send_message_returns_the_reply_with_usage_and_cost(
    client, session_id, fake_openai
):
    response = client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["role"] == "assistant"
    assert body["prompt_tokens"] == 100
    assert body["completion_tokens"] == 10
    # 100 in and 10 out on gpt-4o-mini: 100 * 0.15/1M + 10 * 0.60/1M.
    assert Decimal(body["cost"]) == Decimal("0.000021")


def test_a_message_can_pick_its_own_model(client, session_id, fake_openai):
    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "hi", "model": "gpt-4.1-mini"},
    )

    body = response.json()
    assert fake_openai[0]["model"] == "gpt-4.1-mini"
    assert body["model"] == "gpt-4.1-mini"
    # Priced by gpt-4.1-mini's rates, not the session's cheaper default.
    assert Decimal(body["cost"]) == Decimal("0.000056")


def test_the_session_model_is_not_changed_by_one_message(
    client, session_id, fake_openai
):
    client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "hi", "model": "gpt-4.1-mini"},
    )
    client.post(f"/sessions/{session_id}/messages", json={"content": "again"})

    assert fake_openai[1]["model"] == "gpt-4o-mini"


def test_unsupported_model_is_refused_before_calling_openai(
    client, session_id, fake_openai
):
    response = client.post(
        f"/sessions/{session_id}/messages", json={"content": "hi", "model": "gpt-nope"}
    )

    assert response.status_code == 400
    assert "gpt-4o-mini" in response.json()["error"]["message"]
    # The point of checking first: no request was paid for.
    assert fake_openai == []


def test_the_resolved_model_is_recorded_next_to_the_alias(
    client, session_id, fake_openai
):
    # Priced by the alias, but what actually served the request is kept too.
    body = client.post(
        f"/sessions/{session_id}/messages", json={"content": "hi"}
    ).json()

    assert body["message"]["model"] == "gpt-4o-mini"
    assert body["message"]["resolved_model"] == "gpt-4o-mini-2026-01-01"


def test_session_detail_carries_history_and_totals_together(
    client, session_id, fake_openai
):
    client.post(f"/sessions/{session_id}/messages", json={"content": "one"})
    client.post(f"/sessions/{session_id}/messages", json={"content": "two"})

    body = client.get(f"/sessions/{session_id}").json()

    assert [m["role"] for m in body["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert body["total_prompt_tokens"] == 200
    assert body["total_completion_tokens"] == 20
    assert Decimal(body["total_cost"]) == Decimal("0.000042")


def test_unknown_session_is_a_404(client):
    response = client.get("/sessions/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_empty_content_is_refused(client, session_id):
    response = client.post(f"/sessions/{session_id}/messages", json={"content": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_a_malformed_id_is_refused(client):
    response = client.get("/sessions/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_an_openai_failure_becomes_a_502(client, session_id, failing_openai):
    response = client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_a_failed_call_leaves_nothing_behind(client, session_id, failing_openai):
    # The question must not be stored on its own: the next request would replay
    # it as context and the conversation would read as if it went unanswered.
    client.post(f"/sessions/{session_id}/messages", json={"content": "hi"})

    assert client.get(f"/sessions/{session_id}").json()["messages"] == []
