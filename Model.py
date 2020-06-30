# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:59:38 2020

@author: Sourav
"""

import pandas as pd
#import seaborn as sns

data = pd.read_csv('train.csv')
un = data.nunique()
data = data.drop(['building_id'], axis = 1)


col = []
for i in data:
    if data.dtypes[i] == 'object':
        data[i] = data[i].astype('category')
        col.append(i)
        

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

X = data.drop(['damage_grade'], axis = 1)
Y = data['damage_grade'] 

X = pd.get_dummies(X, drop_first = True)

from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import metrics
import time

train_x, test_x, train_y, test_y = tts(X, Y, stratify=Y,
                                                     random_state=1)

clf = GBC(max_depth = 10, n_estimators = 300, warm_start = True,
          random_state = 104)

start = time.time()
clf.fit(train_x, train_y)
stop = time.time()

y_pred = clf.predict(test_x)
    
# Making the Confusion Matrix
cm = metrics.confusion_matrix(test_y, y_pred)
print(stop-start)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

test = pd.read_csv('test_values.csv')

test = test[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
       'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage',
       'land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other',
       'legal_ownership_status', 'count_families', 'has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel',
       'has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry',
       'has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other']]

col_id = pd.read_csv('test_values.csv', usecols = ['building_id'])

test = pd.get_dummies(test, drop_first = True)

preds = clf.predict(test)

out = pd.DataFrame(preds, columns = ['damage_grade'])

file = pd.concat([col_id, out], axis = 1)

file.to_csv('Output/GBC10.csv', index=False)