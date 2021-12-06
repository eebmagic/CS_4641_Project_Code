import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

class DataSet:
    def __init__(self, zillowSavableCols=['December 2015'], normalize_x=True, normalize_y=True):
        self.census = self.loadCensus()
        self.zillow = self.loadZillow()

        # Remove counties from census data that don't appear in zillow data
        zillowCounties = set(self.zillow['Full County Name'])
        selection = self.census['Full County Name'].isin(zillowCounties)
        self.census = self.census[selection]

        # Remove extra counties in zillow data
        censusCounties = set(self.census['Full County Name'])
        selection = self.zillow['Full County Name'].isin(censusCounties)
        self.zillow = self.zillow[selection]

        # Sort both sets so that rows match up
        self.census = self.census.sort_values(by=['Full County Name'])
        self.zillow = self.zillow.sort_values(by=['Full County Name'])

        # Drop columns with strings
        censusDropableCols = ['State', 'CensusId', 'County', 'Full County Name']
        self.census = self.census.drop(censusDropableCols, axis=1)
        self.zillow = self.zillow[zillowSavableCols]
        self.zillow_actual = np.array(self.zillow.values.tolist())

        self.dataColumns = self.census.columns
        self.targetColumns = self.zillow.columns

        # Normalize data
        self.census = self.census.to_numpy()
        self.zillow = self.zillow.to_numpy()
        if normalize_x:
            scaler = StandardScaler().fit(self.census)
            self.census = scaler.transform(self.census)
        if normalize_y:
            scaler = StandardScaler().fit(self.zillow)
            self.zillow = scaler.transform(self.zillow)


    def loadCensus(self):
        census = pd.read_csv('data/kaggle_us_census_demographic_data/acs2015_county_data.csv')

        # Add column with "County, State" naming scheme
        fullNames = census[['County', 'State']].apply(', '.join, axis=1)
        census['Full County Name'] = fullNames

        return census


    def loadZillow(self):
        zillow = pd.read_csv('data/zillow_rebuilt/zillow_rebuilt.csv')
        zillow = zillow.drop(['Unnamed: 0'], axis=1)

        return zillow


if __name__ == '__main__':
    d = DataSet()

    print('Zillow data:')
    print(d.zillow)
    print(d.zillow.min(), d.zillow.max())
    print(d.zillow.mean(), d.zillow.std())
    print(d.zillow.shape)
    print(type(d.zillow))

    print('Census data:')
    print(d.census)
    print(d.census.min(), d.census.max())
    print(d.census.mean(), d.census.std())
    print(d.census.shape)
    print(type(d.census))

    print(d.dataColumns)
    print(d.targetColumns)
