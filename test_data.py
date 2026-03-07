import pandas as pd

data = pd.read_excel("data/detailed_food_shelf_life_dataset.xlsx")

print(data.head())
print(data.columns)