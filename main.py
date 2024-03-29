import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('nba.csv')
data = data.dropna()
salaries = data['Salary']
data.drop(['Name', 'Team', 'Number', 'Salary'], axis=1, inplace=True)

def newHeight(height):
    if len(height) == 3:
        return int(height[0])*12 + int(height[2])
    return int(height[0])*12 + int(height[2]) + int(height[3])
data['Height'] = [newHeight(h) for h in data['Height']]

def newPosition(position):
    if position == 'SF':
        return 1
    elif position == 'SG':
        return 2
    elif position == 'C':
        return 3
    elif position == 'PF':
        return 4
    elif position == 'PG':
        return 5
data['Position'] = [newPosition(p) for p in data['Position']]

college_dict = {}
college_index = 1
newCollege = []
for c in data['College']:
    if c not in college_dict:
        college_dict[c] = college_index
        college_index += 1
    newCollege.append(college_dict[c])
data['College'] = newCollege

train_X, test_X, train_y, test_y = train_test_split(data, salaries, random_state=0)
model = XGBRegressor(random_state=0, n_estimators=120, learning_rate=0.03)
model.fit(train_X, train_y)
predictions = model.predict(test_X)
mae = mean_absolute_error(predictions, test_y)
print(mae) # 3369272.133241758
