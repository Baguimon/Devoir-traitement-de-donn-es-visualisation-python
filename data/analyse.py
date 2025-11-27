import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_raw = pd.read_csv("result.csv")
print(df_raw.head())
print(df_raw.info())
print(df_raw.memory_usage(deep=True).sum() / 1024, "Ko")