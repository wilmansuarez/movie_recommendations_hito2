import pandas as pd

def load_ratings(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def get_train_test_split(df: pd.DataFrame, test_ratio=0.2):
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_ratio, random_state=42)

