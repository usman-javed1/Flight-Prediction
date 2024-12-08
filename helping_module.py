
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import holidays
import pandas as pd
import numpy as np

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, year=2015):
        self.year = year
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        date = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
        df['WEEK'] = date.dt.isocalendar().week
        df['ISO_YEAR'] = date.dt.isocalendar().year
        week_period = 52
        df['Week_sin'] = np.sin(2 * np.pi * df['WEEK'] / week_period)
        period = 31
        df['Month_sin'] = np.sin(2 * np.pi * df['MONTH'] / period)
        df['Month_cos'] = np.cos(2 * np.pi * df['MONTH'] / period)
        df['Week_cos'] = np.cos(2 * np.pi * df['WEEK'] / week_period)
        df['IS_WEEKEND'] = date.dt.dayofweek.isin([5, 6]).astype(int)
        us_holidays = holidays.UnitedStates(years=self.year)
        holiday_dates = pd.to_datetime(list(us_holidays.keys()))
        df['IS_HOLIDAY'] = date.isin(holiday_dates).astype(int)
        df['DAY_BEFORE_HOLIDAY'] = date.isin(holiday_dates - pd.Timedelta(days=1)).astype(int)
        df['DAY_AFTER_HOLIDAY'] = date.isin(holiday_dates + pd.Timedelta(days=1)).astype(int)
        df['HOLIDAY_NAME'] = date.map(us_holidays).fillna("")
        
        return df


class DropFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df.drop(columns=['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'WEEK' ], inplace=True)
        return df


# cat_Encoder = ColumnTransformer(
#     [('categorical_encoder', OneHotEncoder(sparse_output=False), [11, 12, 13])],
#     remainder='passthrough'
# ).set_output(transform='pandas')

# all_categories = [
#     ["New Year's Day", 'Memorial Day', 'Independence Day', 'Independence Day (observed)', 
#      'Labor Day', 'Veterans Day', 'Thanksgiving', 'Christmas Day', 
#      'Martin Luther King Jr. Day', "Washington's Birthday"]
# ]

# holiday_encoder = ColumnTransformer(
#     transformers=[
#         ('categorical_encoder', OneHotEncoder(sparse_output=False, categories=all_categories,handle_unknown='ignore'), [-1])
#     ],
#     remainder='passthrough'
# ).set_output(transform='pandas')

# num_Encoder = ColumnTransformer(
#     [('numrical_encoder', StandardScaler(), list(range(3, 14)) )],
#     remainder='passthrough'
# ).set_output(transform='pandas')