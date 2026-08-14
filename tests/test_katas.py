from katas.bank import all_hand
from sandbox.runner import run_tests


def test_thirty_hand_katas():
    katas = all_hand()
    assert len(katas) == 30
    ids = [k.id for k in katas]
    assert len(ids) == len(set(ids))
    for k in katas:
        vis = run_tests(k.solution, k.tests)
        hid = run_tests(k.solution, k.hidden_tests)
        assert vis.ok, f"{k.id} visible {vis.summary()}"
        assert hid.ok, f"{k.id} hidden {hid.summary()}"
        assert k.hidden_tests, k.id
