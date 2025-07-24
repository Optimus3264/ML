import pandas as pd

cars = pd.read_csv('newcar_data.csv')
ohe_cars = pd.get_dummies(cars[['car']])

#print(ohe_cars.to_string())

ohe_cars = ohe_cars.astype(int)

print(ohe_cars.to_string())