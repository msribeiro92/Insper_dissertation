from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from pytrends.request import TrendReq

data_path = Path(__file__).parent / 'Data'


def select_cses_columns(cses_df):

    selector = {
        # Identification
        'IMD1008_YEAR': 'Year',
        'IMD1006_NAM': 'Country',
        'IMD1006_UNAlpha2': 'Code',

        # Political information - micro
        'IMD3100_LR_CSES': 'LR_cses_scale',  # VOTE CHOICE: CURRENT MAIN ELECTION - VOTE CHOICE LINKED WITH CSES COLLABORATOR EXPERT JUDGMENT L - R
        'IMD3100_LR_MARPOR': 'LR_rile',  # 'VOTE CHOICE: CURRENT MAIN ELECTION - VOTE CHOICE LINKED WITH MARPOR/CMP RILE

        'IMD3002_LR_CSES': 'LR_cses_cat',  # 'VOTE FOR LEFTIST/CENTER/RIGHTIST - CSES'
        'IMD3002_IF_CSES': 'pp_family',  # 'VOTE CHOICE BY IDEOLOGICAL FAMILY CLASSIFICATION - CSES'

        # Political information - macro

        # Demographic controls
        'IMD2001_2': 'age',  # AGE OF RESPONDENT (IN CATEGORIES)
        'IMD2003': 'education',  # EDUCATION
        'IMD2004': 'marital_status',  # MARITAL STATUS
        'IMD2005_1': 'religious_attendance',  # RELIGIOUS SERVICES ATTENDANCE
        'IMD2006': 'income',  # HOUSEHOLD
        'IMD2007': 'rural_urban',  # RURAL OR URBAN RESIDENCE
        # 'IMD2010': 'race',  # RACE
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

    selected_df = cses_df.rename(columns=selector)[[*selector.values()]]

    return selected_df


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
        # 'race': {
        #     '96': np.NaN,
        #     '97': np.NaN,
        #     '98': np.NaN,
        #     '99': np.NaN,
        # },
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
        'LR_cses_scale': {
            '97': np.NaN,
            '99': np.NaN,
        },
        'LR_rile': {
            '999': np.NaN,
        },
        'gdp_growth_t': {
            '99': np.NaN,
        },
        'gdp_growth_t_1': {
            '99': np.NaN,
        },
        'gdp_growth_t_2': {
            '99': np.NaN,
        },
        'gdp_t': {
            '999999': np.NaN,
        },
        'gdp_t_1': {
            '999999': np.NaN,
        },
        'gdp_t_2': {
            '999999': np.NaN,
        },
        'unem_t': {
            '999': np.NaN,
        },
        'unem_t_1': {
            '999': np.NaN,
        },
        'unem_t_2': {
            '999': np.NaN,
        },
        'hdi_t': {
            '999': np.NaN,
        },
        'hdi_t_1': {
            '999': np.NaN,
        },
        'hdi_t_2': {
            '999': np.NaN,
        },
        'inf_t': {
            '99999': np.NaN,
        },
        'inf_t_1': {
            '99999': np.NaN,
        },
        'inf_t_2': {
            '99999': np.NaN,
        },
    }

    for column, treatment in column_treatments.items():
        cses_df[column] = cses_df[column].transform(lambda x: treatment.get(str(int(x)), x))

    # Simple processing

    # Harmonize country codes with other datasets
    cses_df['Code_2'] = cses_df['Code']
    cses_df['Code'] = cses_df['Code'].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3)

    # Re-label LR_cses_cat
    pp_cses_dict = {
        '1': -1,
        '2': -0,
        '3': 1,
        '9': np.NaN
    }
    cses_df['LR_cses_cat'] = cses_df['LR_cses_cat'].transform(lambda x: pp_cses_dict[str(x)])

    # Re-scale for compatibility with LR_rile
    # cses_df['LR_cses_scale'] = cses_df['LR_cses_scale'] * 20 - 100


def parse_cses_imd():

    # Read file
    full_file_name = data_path / 'cses_imd.csv'
    raw_df = pd.read_csv(full_file_name)

    # Select columns
    df = select_cses_columns(raw_df)

    # Simple processing of raw data
    treat_cses_columns(df)

    return df


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

    columns_to_drop = list(set(df.columns) - {'Year', 'Code', 'ic', 'mc'})
    df = df.drop(columns=columns_to_drop)

    df = df.replace('..', np.NaN)

    add_lagged_columns(df, 'ic', 2)
    add_lagged_columns(df, 'mc', 2)

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


def parse_owid():

    full_file_name = data_path / 'our_world_in_data/technology-adoption-by-households-in-the-united-states.csv'
    df = pd.read_csv(full_file_name)

    parsed_df = pd.DataFrame()
    parsed_df['Year'] = df[df['Entity'] == 'Social media usage']['Year']
    parsed_df['sm_US'] = df[df['Entity'] == 'Social media usage']['Technology Diffusion (Comin and Hobijn (2004) and others)']

    return parsed_df


def encode_categorical_features(df, categorical_columns):

    encoded_measures_list = []

    grouped_df = df.groupby(['Code', 'Year'])

    for name, group in grouped_df:
        for column in categorical_columns:
            args1 = {column+'_cses': pd.NamedAgg(column='LR_cses_scale', aggfunc='mean')}
            cses_encoded_feature = group.groupby(column).agg(**args1).reset_index()
            #cses_encoded_feature = df.groupby(column).agg(**args1).reset_index()
            group = group.merge(cses_encoded_feature, how='left', on=column)
            args2 = {column+'_rile': pd.NamedAgg(column='LR_rile', aggfunc='mean')}
            rile_encoded_feature = group.groupby(column).agg(**args2).reset_index()
            #rile_encoded_feature = df.groupby(column).agg(**args2).reset_index()
            group = group.merge(rile_encoded_feature, how='left', on=column)

            encoded_measures_list.append(group)

    encoded_measures_df = pd.concat(encoded_measures_list)

    return encoded_measures_df


def calculate_polarization_measures(df):

    # Calculate polarization: variance, std
    polarization_measures_df = df.groupby(['Code', 'Year']).agg(
        var_LR_cses_scale=pd.NamedAgg(column='LR_cses_scale', aggfunc='var'),
        var_LR_rile=pd.NamedAgg(column='LR_rile', aggfunc='var'),
        std_LR_cses_scale=pd.NamedAgg(column='LR_cses_scale', aggfunc='std'),
        std_LR_rile=pd.NamedAgg(column='LR_rile', aggfunc='std'),
        mean_LR_cses_scale=pd.NamedAgg(column='LR_cses_scale', aggfunc='mean'),
        mean_LR_rile=pd.NamedAgg(column='LR_rile', aggfunc='mean'),
    )

    return polarization_measures_df


def aggregate_data(df, categorical_columns, other_columns):

    args = {}
    for column in other_columns:
        args[column] = pd.NamedAgg(column=column, aggfunc='first')

    for column in categorical_columns:
        args[column+'_cses'] = pd.NamedAgg(column=column+'_cses', aggfunc='mean')
        args[column+'_rile'] = pd.NamedAgg(column=column+'_rile', aggfunc='mean')

    aggregated_df = df.groupby(['Code', 'Year']).agg(**args)

    return aggregated_df


def add_lagged_columns(df, column_name, nlags):

    column = df[column_name]
    for i in range(nlags):
        shifted_column = column.shift(i+1)
        shifted_column_name = column_name + f"_t_{i+1}"
        df[shifted_column_name] = shifted_column


def interest_by_country_year(pytrend, country, keyword, nlags, cutoffs):

    try:
        pytrend.build_payload(kw_list=[keyword], timeframe='all', geo=country)
        interest_over_time_df = pytrend.interest_over_time()

        # Aggregate by year
        interest_by_year_mean_series = interest_over_time_df[keyword].groupby(interest_over_time_df.index.year).agg('mean')
        interest_by_year_mean_series.rename(keyword)

        interest_by_year_max_series = interest_over_time_df[keyword].groupby(interest_over_time_df.index.year).agg('max')
        df_keyword = interest_by_year_mean_series.to_frame()
        for cutoff in cutoffs:
            max_value = 0
            temp_series = interest_by_year_max_series.copy()
            for i, value in interest_by_year_max_series.iteritems():
                if value >= cutoff:
                    max_value = cutoff
                if max_value == cutoff:
                    temp_series[i] = 1
                else:
                    temp_series[i] = 0

            colum_name = keyword + '_bin_' + str(cutoff)
            df_keyword[colum_name] = temp_series
            add_lagged_columns(df_keyword, colum_name, nlags)

    except Exception as e:
        raise e

    return df_keyword


def get_social_media_data(df, keywords, nlags, cutoffs):

    pytrend = TrendReq()
    countries = np.unique(df['Code_2'])
    interest_by_country = []

    for country in countries:
        interest_by_keywords_list = []
        for keyword in keywords:
            interest_by_keywords_list.append(interest_by_country_year(pytrend, country, keyword, nlags, cutoffs))

        interest_by_keywords = pd.concat(interest_by_keywords_list, axis=1)
        interest_by_keywords['Year'] = interest_by_keywords.index
        interest_by_keywords['Code_2'] = country

        interest_by_country.append(interest_by_keywords)

    interest_by_country_df = pd.concat(interest_by_country)

    return interest_by_country_df


def generate_dataset():

    # Initial dataset
    cses_imd_df = parse_cses_imd()

    wb_df = parse_wb_data()
    wb_df = wb_df.set_index(keys=['Code', 'Year'])

    eiu_df = parse_eiu()
    eiu_df = eiu_df.set_index(keys=['Code', 'Year'])

    owid_df = parse_owid()
    owid_df = owid_df.set_index(keys=['Year'])

    merged_df = cses_imd_df.join(eiu_df, ['Code', 'Year'])
    merged_df = merged_df.join(wb_df, ['Code', 'Year'])
    merged_df = merged_df.join(owid_df, ['Year'])

    # Remove invalid entries
    merged_df = merged_df.dropna(subset=['LR_cses_scale', 'LR_rile']).reset_index(drop=True)

    merged_df.to_csv("merged_data.csv", index=False)

    # Encode categorical features
    categorical_columns = [
        'age',
        'education',
        'marital_status',
        'religious_attendance',
        'income',
        'rural_urban',
        # 'race',
        'number_household',
        'employment',
    ]
    encoded_df = encode_categorical_features(merged_df, categorical_columns)

    # Aggregate by country and year
    other_columns = [
        'Code_2',
        'ic',
        'ic_t_1',
        'ic_t_2',
        'mc',
        'mc_t_1',
        'mc_t_2',
        'gdp_growth_t',
        'gdp_growth_t_1',
        'gdp_growth_t_2',
        'gdp_t',
        'gdp_t_1',
        'gdp_t_2',
        'unem_t',
        'unem_t_1',
        'unem_t_2',
        'hdi_t',
        'hdi_t_1',
        'hdi_t_2',
        'inf_t',
        'inf_t_1',
        'inf_t_2',
        'VA',
        'PV',
        'GE',
        'RQ',
        'RL',
        'CC'
    ]
    aggregated_df = aggregate_data(encoded_df, categorical_columns, other_columns)

    # Add polarization measures
    polarization_measures_df = calculate_polarization_measures(encoded_df)

    aggregated_df = aggregated_df.merge(polarization_measures_df, left_index=True, right_index=True)
    aggregated_df = aggregated_df.reset_index()

    aggregated_df.to_csv("aggregated_data.csv", index=False)

    # Add social media data from Google Trends
    social_media_keywords = ['facebook', 'twitter', 'youtube']
    social_media_data = get_social_media_data(aggregated_df, social_media_keywords, 2, [10, 30, 50, 70, 90])
    social_media_df = aggregated_df.merge(social_media_data, how='left', on=['Year', 'Code_2'])
    social_media_df = social_media_df.dropna(subset=['facebook', 'twitter', 'youtube']).reset_index(drop=True)

    social_media_df.to_csv("social_media_data.csv", index=False)


generate_dataset()
print('end')
