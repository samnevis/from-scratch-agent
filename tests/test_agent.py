from agent.loop import parse_tool_call, run_episode
from agent.policies import gold_policy, recovery_policy
from katas.bank import all_hand, by_id


def test_parse_tool_call():
    raw = 'noise {"tool":"read_task","args":{}} more'
    call = parse_tool_call(raw)
    assert call["tool"] == "read_task"


def test_parse_def_as_write_solution():
    call = parse_tool_call("def add(a, b):\n    return a + b\n")
    assert call["tool"] == "write_solution"
    assert "def add" in call["args"]["code"]


def test_gold_solves_all_hand():
    for k in all_hand():
        st = run_episode(k, gold_policy)
        assert st.hidden_ok(), k.id


def test_recovery_solves_one():
    k = by_id("hand_002")
    st = run_episode(k, recovery_policy)
    assert st.hidden_ok()
    assert any("passed" in (t.get("obs") or "") or "broken" in str(t) for t in st.transcript)
