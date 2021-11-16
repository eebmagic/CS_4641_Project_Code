import pandas as pd
from tqdm import tqdm

state_names = {
    'AL': 'Alabama', 'NE': 'Nebraska', 'AK': 'Alaska', 'NV': 'Nevada',
    'AZ': 'Arizona', 'NH': 'New Hampshire', 'AR': 'Arkansas',
    'NJ': 'New Jersey', 'CA': 'California', 'NM': 'New Mexico',
    'CO': 'Colorado', 'NY': 'New York', 'CT': 'Connecticut',
    'NC': 'North Carolina', 'DE': 'Delaware', 'ND': 'North Dakota',
    'DC': 'District of Columnbia', 'OH': 'Ohio', 'FL': 'Florida',
    'OK': 'Oklahoma', 'GA': 'Georgia', 'OR': 'Oregon', 'HI': 'Hawaii',
    'PA': 'Pennsylvania', 'ID': 'Idaho', 'PR': 'Puerto Rico',
    'IL': 'Illinois', 'RI': 'Rhode Island', 'IN': 'Indiana',
    'SC': 'South Carolina', 'IA': 'Iowa', 'SD': 'South Dakota',
    'KS': 'Kansas', 'TN': 'Tennessee', 'KY': 'Kentucky',
    'TX': 'Texas', 'LA': 'Louisiana', 'UT': 'Utah', 'ME': 'Maine',
    'VT': 'Vermont', 'MD': 'Maryland', 'VA': 'Virginia',
    'MA': 'Massachusetts', 'VI': 'Virgin Islands', 'MI': 'Michigan',
    'WA': 'Washington', 'MN': 'Minnesota', 'WV': 'West Virginia',
    'MS': 'Mississippi', 'WI': 'Wisconsin', 'MO': 'Missouri',
    'WY': 'Wyoming', 'MT': 'Montana'
}

includedLabels = [
    # 'County', 'State', 'Population Rank', 'Full County Name'
    'Full County Name'
]

includedColumns = [
    'January 2015', 'February 2015', 'March 2015', 'April 2015',
    'May 2015', 'June 2015', 'July 2015', 'August 2015', 'September 2015',
    'October 2015', 'November 2015', 'December 2015'
]

def mergeRows(df, includedLabels, includedColumns):
    '''
    Take in a selection of the data and return a dictionary that can
    be appended to a dataframe.

    Include singular values in includedLabels.
    Include averages of items listed in includedColumns.
    '''
    assert len(df['Full County Name'].unique()) == 1, 'need only one county'

    out = {}
    for label in includedLabels:
        assert len(df[label].unique()) == 1, f'multiple values found for {label}'
        out[label] = df[label].unique()[0]

    for column in includedColumns:
        out[column] = df[column].mean()

    return out


zillow = pd.read_csv('../kaggle_zillow_rent_index/pricepersqft.csv')

# Replace state Abreviations with full names
zillow = zillow.replace({'State': state_names})

# Add column with "County, State" naming scheme
fullNames = zillow[['County', 'State']].apply(', '.join, axis=1)
zillow['Full County Name'] = fullNames

# Merge divided counties
merged = pd.DataFrame()

for name in tqdm(zillow['Full County Name'].unique()):
    rows = zillow[zillow['Full County Name'] == name]
    rebuilt = mergeRows(rows, includedLabels, includedColumns)

    merged = merged.append(rebuilt, ignore_index=True)

print(merged[['Full County Name', 'December 2015']])
print(len(merged))

merged.to_csv('zillow_rebuilt.csv')
print('FINISHED saving to file')
