import pandas as pd
from pytrends.request import TrendReq


def interest_by_country_year(pytrend, country, keyword):

    try:
        pytrend.build_payload(kw_list=[keyword], timeframe='all', geo=country)
        interest_over_time_df = pytrend.interest_over_time()

        # Aggregate by year
        interest_by_year_series = interest_over_time_df[keyword].groupby(interest_over_time_df.index.year).agg('mean')
        interest_by_year_series.rename(keyword)

    except Exception as e:
        raise e

    return interest_by_year_series


countries = ['US']
kw_list = ['facebook']

pytrend = TrendReq()

for kw in kw_list:
    for country in countries:
        interest_by_country_year(pytrend, country, kw)
