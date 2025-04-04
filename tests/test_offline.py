from recommender.offline import compute_rmse

def test_rmse():
    preds = [3.5, 4.0, 2.0]
    trues = [3.0, 4.0, 2.0]
    assert compute_rmse(preds, trues) == 0.28867513459481287
