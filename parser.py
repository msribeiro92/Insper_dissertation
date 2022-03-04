from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

data_path = Path(__file__).parent / 'Data'


def treat_cses_columns(cses_df):

    # Undefined categories treatment
    column_treatments = {
        'age': {
            '9997': np.NaN,
            '9998': np.NaN,
            '9999': np.NaN
        },
        'education': {
            '7': np.NaN,
            '8': np.NaN,
            '9': np.NaN
        },
        'marital_status': {
            '5': np.NaN,
            '7': np.NaN,
            '8': np.NaN,
            '9': np.NaN
        },
        'religious_attendance': {
            '7': np.NaN,
            '8': np.NaN,
            '9': np.NaN
        },
        'income': {
            '7': np.NaN,
            '8': np.NaN,
            '9': np.NaN
        },
        'rural_urban': {
            '7': np.NaN,
            '8': np.NaN,
            '9': np.NaN
        },
        'race': {
            '96': np.NaN,
            '97': np.NaN,
            '98': np.NaN,
            '99': np.NaN,
        },
        'number_household': {
            '97': np.NaN,
            '98': np.NaN,
            '99': np.NaN,
        },
        'employment': {
            '97': np.NaN,
            '98': np.NaN,
            '99': np.NaN,
        },
    }

    for column, treatment in column_treatments.items():
        cses_df[column] = cses_df[column].transform(lambda x: treatment.get(str(x), x))

    # Simple processing
    cses_df['Code'] = cses_df['Code'].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3)


def parse_cses_imd():

    full_file_name = data_path / 'cses_imd.csv'
    df = pd.read_csv(full_file_name)

    selector = {
        # Identification
        'IMD1008_YEAR': 'Year',
        'IMD1006_NAM': 'Country',
        'IMD1006_UNAlpha2': 'Code',

        # Political information - micro
        'IMD3002_LR_CSES': 'LR_cses_cat',  # 'VOTE FOR LEFTIST/CENTER/RIGHTIST - CSES'
        'IMD3100_LR_CSES': 'LR_cses_scale',  # VOTE CHOICE: CURRENT MAIN ELECTION - VOTE CHOICE LINKED WITH CSES COLLABORATOR EXPERT JUDGMENT L - R
        'IMD3100_LR_MARPOR': 'LR_rile',  # 'VOTE CHOICE: CURRENT MAIN ELECTION - VOTE CHOICE LINKED WITH MARPOR/CMP RILE


        'IMD3002_IF_CSES': 'pp_family',  # 'VOTE CHOICE BY IDEOLOGICAL FAMILY CLASSIFICATION - CSES'

        # Political information - macro

        # Demographic controls
        'IMD2001_2': 'age',  # AGE OF RESPONDENT (IN CATEGORIES)
        'IMD2003': 'education',  # EDUCATION
        'IMD2004': 'marital_status',  # MARITAL STATUS
        'IMD2005_1': 'religious_attendance',  # RELIGIOUS SERVICES ATTENDANCE
        'IMD2006': 'income',  # HOUSEHOLD
        'IMD2007': 'rural_urban',  # RURAL OR URBAN RESIDENCE
        'IMD2010': 'race',  # RACE
        'IMD2012_1': 'number_household',   # NUMBER IN HOUSEHOLD IN TOTAL
        'IMD2014': 'employment',  # CURRENT EMPLOYMENT STATUS

        # Macroeconomic controls
        'IMD5052_1': 'gdp_growth_t',  # GDP GROWTH ANNUAL % (WORLD BANK) - TIME T
        'IMD5052_2': 'gdp_growth_t_1',  # GDP GROWTH ANNUAL % (WORLD BANK) - TIME T-1 YEAR
        'IMD5052_3': 'gdp_growth_t_2',  # GDP GROWTH ANNUAL % (WORLD BANK) - TIME T-2 YEARS
        'IMD5053_1': 'gdp_t',  # GDP PER CAPITA, PPP (WORLD BANK) - TIME T
        'IMD5053_2': 'gdp_t_1',  # GDP PER CAPITA, PPP (WORLD BANK) - TIME T-1 YEAR
        'IMD5053_3': 'gdp_t_2',  # GDP PER CAPITA, PPP (WORLD BANK) - TIME T-2 YEARS
        'IMD5054_1': 'unem_t',  # UNEMPLOYMENT, TOTAL (WORLD BANK) - TIME T
        'IMD5054_2': 'unem_t_1',  # UNEMPLOYMENT, TOTAL (WORLD BANK) - TIME T-1 YEAR
        'IMD5054_3': 'unem_t_2',  # UNEMPLOYMENT, TOTAL (WORLD BANK) - TIME T-2 YEARS
        'IMD5055_1': 'hdi_t',  # HUMAN DEVELOPMENT INDEX (UNPD)  - TIME T
        'IMD5055_2': 'hdi_t_1',  # HUMAN DEVELOPMENT INDEX (UNPD)  - TIME T-1 YEAR
        'IMD5055_3': 'hdi_t_2',  # HUMAN DEVELOPMENT INDEX (UNPD)  - TIME T-2 YEARS
        'IMD5056_1': 'inf_t',  # INFLATION, GDP DEFLATOR (ANNUAL %) (WORLD BANK) - TIME T
        'IMD5056_2': 'inf_t_1',  # INFLATION, GDP DEFLATOR (ANNUAL %) (WORLD BANK) - TIME T-1 YEAR
        'IMD5056_3': 'inf_t_2',  # INFLATION, GDP DEFLATOR (ANNUAL %) (WORLD BANK) - TIME T-2 YEARS
        # 'IMD5057_1': 'pop_abs_t',  # POPULATION, TOTAL (WORLD BANK) - TIME T
        # 'IMD5057_2': 'pop_abs_t_1',  # POPULATION, TOTAL (WORLD BANK) - TIME T-1 YEAR
        # 'IMD5057_3': 'pop_abs_t_2',  # POPULATION, TOTAL (WORLD BANK) - TIME T-2 YEARS
    }

    df = df.rename(columns=selector)[[*selector.values()]]

    treat_cses_columns(df)

    # Treat polarization measures
    pp_cses_dict = {
        '1': -1,
        '2': -0,
        '3': 1,
        '9': np.NaN
    }
    df['pp_cses'] = df['pp_cses'].transform(lambda x: pp_cses_dict[str(x)])

    return df


def parse_eiu():

    full_file_name = data_path / 'EIU.xlsx'
    xl = pd.ExcelFile(full_file_name)
    df = xl.parse(sheet_name='Data', header=[0, 1])

    columns0 = df.columns.get_level_values(0).array
    columns1 = df.columns.get_level_values(1).array
    parsed_df = pd.DataFrame()
    sliced_df = pd.DataFrame()
    year = None
    for i in range(2, len(columns0)):
        if year is None or not year == columns1[i]:
            year = columns1[i]
            parsed_df = parsed_df.append(sliced_df)
            sliced_df = pd.DataFrame()
            sliced_df['Country'] = df[columns0[0]][columns1[0]]
            sliced_df['Code'] = df[columns0[1]][columns1[1]]
            sliced_df['Year'] = year
        column_name = ''
        if 'VA' in columns0[i]:
            column_name = 'VA'  # 'Voice and Accountability'
        if 'PV' in columns0[i]:
            column_name = 'PV'  # 'Political Stability and Absence of Violence'
        if 'GE' in columns0[i]:
            column_name = 'GE'  # 'Government Effectiveness'
        if 'RQ' in columns0[i]:
            column_name = 'RQ'  # 'Regulatory Quality'
        if 'RL' in columns0[i]:
            column_name = 'RL'  # 'Rule of Law'
        if 'CC' in columns0[i]:
            column_name = 'CC'  # 'Control of Corruption'
        sliced_df[column_name] = df[columns0[i]][columns1[i]]
    parsed_df = parsed_df.append(sliced_df)
    parsed_df = parsed_df.dropna(subset=['Country'])

    columns_to_drop = ['Country']
    parsed_df = parsed_df.drop(columns=columns_to_drop)

    return parsed_df


def parse_wb_data():

    full_file_name = data_path / 'Data_Extract_From_World_Development_Indicators.xlsx'
    xl = pd.ExcelFile(full_file_name)
    df = xl.parse(sheet_name='Data')

    df = df.rename(
        columns={
            'Time': 'Year',
            'Country Name': 'Country',
            'Country Code': 'Code',
            'Individuals using the Internet (% of population) [IT.NET.USER.ZS]': 'ic',
            'Mobile cellular subscriptions (per 100 people) [IT.CEL.SETS.P2]': 'mc',
        }
    )

    df = df.dropna(subset=['Code'])
    df = df.astype({'Year': int})

    columns_to_drop = list(set(df.columns) - set(['Year', 'Code', 'ic', 'mc']))
    df = df.drop(columns=columns_to_drop)

    df = df.replace('..', np.NaN)

    return df


def parse_owid():

    full_file_name = data_path / 'our_world_in_data/technology-adoption-by-households-in-the-united-states.csv'
    df = pd.read_csv(full_file_name)

    parsed_df = pd.DataFrame()
    parsed_df['Year'] = df[df['Entity'] == 'Social media usage']['Year']
    parsed_df['sm_US'] = df[df['Entity'] == 'Social media usage']['Technology Diffusion (Comin and Hobijn (2004) and others)']

    return parsed_df


cses_imd_df = parse_cses_imd()

eiu_df = parse_eiu()
eiu_df = eiu_df.set_index(keys=['Code', 'Year'])

wb_df = parse_wb_data()
wb_df = wb_df.set_index(keys=['Code', 'Year'])

owid_df = parse_owid()
owid_df = owid_df.set_index(keys=['Year'])

merged_df = pd.DataFrame()
merged_df = cses_imd_df.join(eiu_df, ['Code', 'Year'])
merged_df = merged_df.join(wb_df, ['Code', 'Year'])
merged_df = merged_df.join(owid_df, ['Year'])

merged_df.to_csv("merged_data.csv", index=False)

print('end')