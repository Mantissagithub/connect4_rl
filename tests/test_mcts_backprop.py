from mcts_components.backpropogate_value import backpropagate_value


def test_backprop_flips_sign_each_level():
    root = {"total_value": 0.0, "visits": 0}
    child = {"total_value": 0.0, "visits": 0}
    leaf = {"total_value": 0.0, "visits": 0}
    path = [root, child, leaf]

    backpropagate_value(path, 1.0)

    assert leaf["total_value"] == 1.0
    assert child["total_value"] == -1.0
    assert root["total_value"] == 1.0
    assert root["visits"] == child["visits"] == leaf["visits"] == 1
