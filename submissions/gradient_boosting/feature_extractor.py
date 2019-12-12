import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from recordlinkage.preprocessing import clean


class FeatureExtractor(object):
    def __init__(self):
        pass

    def transform(self, X_df):
        return self.preprocessor.transform(X_df)

    def fit(self, X_df, y_array):
        X_encoded = X_df 

        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'), compression='zip', low_memory=False)

        # Zipcodes of prefectures.
        self.big_city_zip = {
            'caen': 14000, 'dijon': 21000, 'strasbourg': 67000, 
            'nantes': 44000, 'amiens': 80000,'orleans': 45000,
            'montpellier': 34000, 'ajaccio': 20000, 'marseille': 13000,
            'limoges': 87000, 'rennes': 35000, 'besancon': 25000,
            'chalonsenchampagne': 51000, 'metz': 57000, 'lille': 59000,
            'bordeaux': 33000, 'rouen': 76000, 'clermontferrand': 63000,
            'toulouse': 31000, 'paris': 75000, 'lyon': 69000, 'poitiers': 86000
        }

        def remove_cedex(x):
            return str(x).replace('cedex', '').replace('ceddex', '').replace(
                'ceedex', '').replace('cededx', '').replace('cadex', '').replace(
                'cdedex', '').replace('vedex', '').replace('crdex', '').replace(
                'dedex', '').replace('edex', '').replace('cdex', '').replace(
                'cedx', '').replace('cdx', '')

        # Cleaning 'City' feature.
        def deep_clean(X, feature='City'):
            return clean(X[feature]).str.replace('[^\w]','').str.replace('[0-9]','').apply(lambda x: remove_cedex(x))

        # Creating a dictionnary with the zipcode of each city in order to have uniform zipcodes accross cities
        # and to fill some NaN values.
        X_clean = X_encoded[['City', 'Zipcode']].copy()
        X_clean['City_clean'] = deep_clean(X_clean, 'City')
        X_clean['Zipcode'] = pd.to_numeric(X_clean['Zipcode'], errors='coerce')
        self.city_zip = X_clean[~X_clean['City_clean'].isin(self.big_city_zip.keys())][['City_clean', 'Zipcode']].dropna()

        aw_clean = award[['incumbent_city', 'incumbent_zipcode']].copy()
        aw_clean['City_clean'] = deep_clean(aw_clean, 'incumbent_city')
        aw_clean['Zipcode'] = pd.to_numeric(aw_clean['incumbent_zipcode'], errors='coerce')
        city_zip_aw = aw_clean[~aw_clean['City_clean'].isin(self.big_city_zip.keys())][['City_clean', 'Zipcode']].dropna()

        # Zipcodes of all the cities.
        self.other_city_zip = pd.concat([self.city_zip, city_zip_aw[~city_zip_aw['City_clean'].isin(self.city_zip['City_clean'])]])
        self.other_city_zip = self.other_city_zip.groupby('City_clean').agg(lambda x: x.value_counts().index[0])['Zipcode'].to_dict()
        self.all_city_zip = {**self.other_city_zip, **self.big_city_zip}

        # Processing the award dataframe
        award.dropna(subset=['awarded','Contract_awarded','incumbent_name'], inplace=True)
        award.drop(['awarded','Contract_awarded'], axis=1, inplace=True)
        award['Year'] = pd.to_datetime(award['Publication_date'], format='%Y-%m-%d').dt.year
        award.drop(['Publication_date'], axis=1, inplace=True)
        award['name'] = deep_clean(award, 'incumbent_name')
        award_num = award[['Year', 'Total_amount', 'Lot', 'number_of_received_bids', 'amount']]
        award_num = award_num.apply(pd.to_numeric, errors='coerce').fillna(award_num.median()) 
        award_sum = award.groupby(['name', 'Year'])[
            ['Total_amount', 'Lot', 'number_of_received_bids', 'amount']].sum()
        award_count = award.groupby(['name', 'Year'])[
            ['Total_amount', 'Lot', 'number_of_received_bids', 'amount']].count()
        award_count.columns = award_count.columns + '_c'
        award_features = pd.concat([award_sum, award_count], axis=1).reset_index()

        # Fiilling zipcode values
        def fill_zipcode(row, all_city_zip=self.all_city_zip):
            if row['City_clean'] in all_city_zip.keys():
                return all_city_zip[row['City_clean']]
            else:
                return row['Zipcode']

        def clean_city_zipcode(X, all_city_zip=self.all_city_zip):
            X_clean = X[['City', 'Zipcode']].copy()
            # City cleaning 
            X_clean['City_clean'] = deep_clean(X_clean, 'City')
            # Zipcode cleaning
            X_clean.loc[pd.notnull(X_clean['City_clean']), 'Zipcode'] = X_clean.loc[
                pd.notnull(X_clean['City_clean']), ['Zipcode', 'City_clean']
            ].apply(lambda row: fill_zipcode(row, all_city_zip), axis=1)
            return X_clean[['City_clean', 'Zipcode']]

        # Creating a new feature using 'City' and 'Zipcode' providing the size and the type of a city.
        def city_size(row, big_city_zip=self.big_city_zip):
            big_cities = ['paris', 'marseille', 'toulouse', 'lyon']
            city, zipcode = row["City_clean"], row["Zipcode"]
            if (pd.notnull(city)) and (city in big_city_zip.keys()):
                if city in big_cities:
                    return city
                else:
                    return 'prefecture'
            else:
                if pd.notnull(zipcode):
                    if str(zipcode)[:2] in ['75', '77', '78', '91', '92', '93', '94', '95']:
                        return 'ile_france'
                    elif zipcode >= 97000:
                        return 'overseas'
                    elif zipcode % 1000 == 0:
                        return 'big_city'
                    elif zipcode % 100 == 0:
                        return 'medium_city'
                    elif zipcode % 10 == 0:
                        return 'small_city'
                    else:
                        return 'very_small_city'
                else:
                    return 'other'

        def process_date(X):
            date = pd.to_datetime(X['Fiscal_year_end_date'], format='%Y-%m-%d')
            return np.c_[date.dt.month, date.dt.day]
        date_transformer = FunctionTransformer(process_date, validate=False)

        def process_APE(X):
            APE = X['Activity_code (APE)'].str[:-1]
            return pd.to_numeric(APE, errors='coerce').values[:, np.newaxis] 
        APE_transformer = FunctionTransformer(process_APE, validate=False)

        def process_zipcode(X, all_city_zip=self.all_city_zip):
            zipcode_nums = pd.to_numeric(clean_city_zipcode(X, all_city_zip)['Zipcode'], errors='coerce')
            return zipcode_nums.values[:, np.newaxis]
        zipcode_transformer = FunctionTransformer(process_zipcode, validate=False)

        def process_city(X, big_city_zip=self.big_city_zip, all_city_zip=self.all_city_zip):
            X_clean = clean_city_zipcode(X, all_city_zip)
            X_clean['Zipcode'] = pd.to_numeric(X_clean['Zipcode'], errors='coerce')
            return X_clean.apply(lambda row: city_size(row, big_city_zip), axis=1).values[:, np.newaxis]
        city_transformer = FunctionTransformer(process_city, validate=False)

        def merge(X):
            X['Name'] = deep_clean(X, 'Name')
            df = pd.merge(X, award_features, left_on=['Name', 'Year'], right_on=['name', 'Year'], how='left')
            return df[['Total_amount', 'Lot', 'number_of_received_bids', 'amount',
                       'Total_amount_c', 'Lot_c', 'number_of_received_bids_c', 'amount_c']]
        merge_transformer = FunctionTransformer(merge, validate=False)

        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))])
 
        date_cols = ['Fiscal_year_end_date']
        num_cols = ['Legal_ID', 'Year', 'Headcount']
        APE_cols = ['Activity_code (APE)']
        zipcode_cols = ['Zipcode', 'City']
        city_cols = ['Zipcode', 'City']
        merge_cols = ['Name', 'Year']
        drop_cols = ['Address', 'Fiscal_year_duration_in_months']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('date', make_pipeline(date_transformer, SimpleImputer(strategy='median')), date_cols),
                ('num', numeric_transformer, num_cols),
                ('APE', make_pipeline(APE_transformer, SimpleImputer(strategy='median')), APE_cols),
                ('zipcode', make_pipeline(zipcode_transformer, SimpleImputer(strategy='median')), zipcode_cols),
                ('city', make_pipeline(city_transformer, SimpleImputer(strategy='most_frequent'), OrdinalEncoder()), city_cols),
                ('merge', make_pipeline(merge_transformer, SimpleImputer(strategy='median')), merge_cols),
                ('drop cols', 'drop', drop_cols),
            ])

        self.preprocessor.fit(X_encoded, y_array)
