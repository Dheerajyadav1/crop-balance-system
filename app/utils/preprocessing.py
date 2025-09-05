# small helpers if you want to share preprocessing logic across modules
import pandas as pd

def load_final_data(path):
    df = pd.read_csv(path)
    return df
