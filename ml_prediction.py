import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random

pd.set_option('display.float_format', lambda x: '%.5f' % x)

def get_dates():
    start_dates = []
    for year in range(1980, 2023):
        for month in ["31/01", "28/02","29/02", "31/03", "30/04", "31/05", "30/06", "31/07", "31/08", "30/09", "31/10", "30/11", "31/12"]:
            date = month + "/" + str(year)
            start_dates.append(date)
    return start_dates

def concat_indicators():
    path = "..//data//predictors//raw"
    indicator_groups = [file for file in os.listdir(path) if file[-4:] != ".csv"]
    dates = get_dates()
    random.shuffle(dates)
    for date in tqdm(dates):
        new_date_format = date[-4:] + "-" + date[-7:-5] + "-" + date[:2]
        res_file = os.path.join("..//data//all_predictors", new_date_format + ".csv")
        if not os.path.isfile(res_file):
            indicator_dfs = []
            for group in tqdm(indicator_groups):
                files = os.listdir(os.path.join(path, group))
                for file in files:
                    indicator = file[:-4]
                    try:
                        ind_df = pd.read_csv(os.path.join(path, group, file), usecols=["SecId", date], low_memory=False)
                    except:
                        try:
                            ind_df = pd.read_csv(os.path.join(path, group, file), usecols=["SecId", new_date_format], low_memory=False)
                        except:
                            continue
                    ind_df.rename(columns={date: indicator, new_date_format: indicator}, inplace=True)
                    indicator_date_df = ind_df[["SecId", indicator]].dropna().set_index("SecId")
                    indicator_dfs.append(indicator_date_df)
            date_df = pd.DataFrame().join(indicator_dfs, how="outer")

            date_df.to_csv(res_file)

def save_features_labels():
    path = "..//data//all_predictors"
    monthly_returns = pd.read_csv("..//data//returns.csv")
    files = os.listdir(path)
    for file in tqdm(files):
            date = file[:-6] + "01"
            res_file = os.path.join("..//data//features", date + ".csv")
        #if not os.path.isfile(res_file):
            next_date = (datetime.strptime(date, "%Y-%m-%d") + relativedelta(months=1)).strftime("%Y-%m-%d")
            indicators = pd.read_csv(os.path.join(path, file), low_memory=False)
            if not indicators.empty:
                try:
                    stock_return_next_df = monthly_returns[["SecId", next_date]]
                    stock_return_next_df.rename(columns={next_date:"ret"}, inplace=True)
                    indicators = pd.merge(indicators, stock_return_next_df, on="SecId",
                                              how="inner")
                    #share_nan = len(indicators["Alpha_1y"].dropna()) / len(indicators)
                    indicators = indicators.replace([np.inf, -np.inf], np.nan).fillna(0).drop_duplicates()
                    for col in indicators.columns:
                        if col not in ["SecId"]:
                            try:
                                indicators[col] = [float(val.replace(",", "")) if type(val) == str else val for val in
                                               indicators[col]]
                            except:
                                indicators = indicators.drop(col, axis=1)
                    indicators.to_csv(res_file, index=False)
                except:
                    pass

def train_model(eval_month):
    path = "..//data//features"
    feature_df = pd.concat([pd.read_csv(os.path.join(path, file), low_memory=False) for file in
        tqdm(os.listdir(path)) if file[:-4] < eval_month])

    feature_df = feature_df.replace(0, np.nan)
    # get rid of rows with too few data
    print(feature_df.count(axis="columns"))
    feature_df["non_zero"] = feature_df.count(axis="columns")
    feature_df["non_zero_share"] = feature_df["non_zero"] / (len(feature_df.columns)-1)
    feature_df = feature_df[feature_df["non_zero_share"]>=0.5].drop(["SecId", "non_zero_share", "non_zero"],
                                                                       axis=1)

    # get rid of columns with too few data

    num_obs = len(feature_df)
    for col in tqdm(feature_df.columns):
        if int(feature_df[col].count()) / num_obs < 0.50:
            feature_df = feature_df.drop(col, axis=1)
        else:
            if max(abs(feature_df[col])) >= 100000000:
                feature_df = feature_df.drop(col, axis=1)

    labels = list(feature_df["ret"])
    feature_df = feature_df.replace([np.nan, np.inf, -np.inf], 0).drop(["ret"],
                                                                       axis=1)

    print(len(feature_df))
    print(list(feature_df.columns))
    rf = RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=1)
    filename = '..//models//basic_rf.save'

    model = rf.fit(feature_df, labels)
    pickle.dump(model, open(filename, 'wb'))


#concat_indicators()
#save_features_labels()
train_model(eval_month="2005-01-01")
