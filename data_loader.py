import pandas as pd

class DataSet:
    def __init__(self):
        self.census = self.loadCensus()
        self.zillow = self.loadZillow()

        # Remove counties from census data that don't appear in zillow data
        zillowCounties = set(self.zillow['Full County Name'])
        selection = self.census['Full County Name'].isin(zillowCounties)
        self.census = self.census[selection]


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
    print(d.census)
    print(d.zillow)
