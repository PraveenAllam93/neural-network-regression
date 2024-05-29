import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def date_extractor(date):

    """
    Takes a date (feature) as input, and gives the month, day, and hour present in that date
    
    Args:
        date (pandas.core.series.Series): The Series of dates which is present Object dtype

    Returns: 
        tuple : returns a tuple with two ojects month, day and one int32 hour 
    """

    date = pd.to_datetime(date)

    month = date.dt.month_name()
    day = date.dt.day_name()
    hour = date.dt.hour

    return month, day, hour

def delivery_time_calculator(created_at, actual_delivery_time):

    """
    Takes two date (feature) as inputs, and gives the difference between those dates in minutes
    
    Args:
        created_at (pandas.core.series.Series): The start date, at which the order is created
        actual_delivery_time (pandas.core.series.Series): The end date, at which the order is delivered

    Returns: 
        delivery_time (pandas.core.series.Series): The difference between created_at and actual_delivery_time in minutes
    """

    created_at = pd.to_datetime(created_at)
    actual_delivery_time = pd.to_datetime(actual_delivery_time)

    delivery_time = (actual_delivery_time - created_at).dt.total_seconds()/60

    return delivery_time

def weekend_info(weekday):

    """
    Takes a weekday (feature) as inputs, and gives whether it is weekend or not
    
    Args:
        weekday (pandas.core.series.Series): The week day, at which the order is created
       

    Returns: 
        is_weekend (pandas.core.series.Series): whether the given day is a weekend or not
    """

    weekend = {'Friday' : 0, 'Tuesday' : 0, 'Monday' : 0, 'Thursday' : 0, 'Sunday' : 1, 'Saturday' : 1,
       'Wednesday' : 1}

    is_weekend = weekday.map(weekend)
    return is_weekend

def filling_nan_values(df, cols):

    """
    Takes a df (dataframe) and cols (missing values present features) as inputs, and gives a df with zero NaN values
    
    Args:
        df (pandas.core.DataFrame): The DataFrame for which you want to fill the NaN values
        cols (list): The columns (features) which got missing values
    Returns: 
        df (pandas.core.DataFrame): a data frame with no missing values
    """

    for col in cols:
        if df[col].dtype != "O":
            max_col = df[col].max()
            df[col] = df[col].fillna(max_col + 1)
        else:
            df[col] = df[col].fillna("other")
    return df  

def delivery_partners(df, given_info, avg, total_onshift_partners = False, total_busy_partners = False, total_outstanding_orders = False):
    if (total_onshift_partners == True):
        total_onshift_partners = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.created_at_hour == df.created_at_hour)]["total_onshift_partners"]
        if np.isnan(total_onshift_partners.iloc[0]):
            return avg
        return total_onshift_partners.iloc[0]
    if (total_busy_partners == True):
        total_busy_partners = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.created_at_hour == df.created_at_hour)]["total_busy_partners"]
        if np.isnan(total_busy_partners.iloc[0]):
            return avg
        return total_busy_partners.iloc[0]
    if (total_outstanding_orders == True):
        total_outstanding_orders = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.created_at_hour == df.created_at_hour)]["total_outstanding_orders"]
        if np.isnan(total_outstanding_orders.iloc[0]):
            return avg
        return total_outstanding_orders.iloc[0]
    return "atleast choose one" 


def preprocessing_before_split(df):

    df["created_at_month"], df["created_at_weekday"], df["created_at_hour"] = date_extractor(df.created_at)
    df["actual_delivery_month"], df["actual_deliveryt_weekday"], df["actual_delivery_hour"] = date_extractor(df.actual_delivery_time)

    df["delivery_time"] = delivery_time_calculator(df.created_at, df.actual_delivery_time)
    df["is_weekend"] = weekend_info(df.created_at_weekday)

    missing_value_cols = ["market_id", "store_primary_category", "order_protocol"]
    df.dropna(subset = ["actual_delivery_time", "actual_delivery_month", "actual_deliveryt_weekday", "actual_delivery_hour"], inplace = True)

    df = filling_nan_values(df, cols = missing_value_cols)
    print(f"remaining cols with missing values = {df.columns[df.isna().sum() > 0]}")

    return df

def outliers_handler(train, test, transform_cols):
    
    for col in transform_cols:
        lb = np.quantile(train[col],0.25)
        ub = np.quantile(train[col],0.75)
        IQR = ub - lb

        upper_limit = ub + 3 * IQR
        lower_limit = lb - 3 * IQR

    train = train[(train[col] >= lower_limit) & (train[col] <= upper_limit)]
    test = test[(test[col] >= lower_limit) & (test[col] <= upper_limit)]

    return train, test

def target_encoding(train, test, cols):

    te = TargetEncoder(cols = cols)
    te.fit(train.drop("delivery_time", axis = 1), train.delivery_time)

    X_train = te.transform(train.drop("delivery_time", axis = 1))
    X_test = te.transform(test.drop("delivery_time", axis = 1))

    return X_train, X_test

def standard_scaling(X_train, X_test):

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_sca = scaler.transform(X_train)
    X_test_sca = scaler.transform(X_test)

    return X_train_sca, X_test_sca


def preprocessing_after_split(df, delivery_partners_avgs, delivery_partners_grps):

    df.loc[df.total_onshift_partners.isna(), "total_onshift_partners"] = df.loc[df.total_onshift_partners.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_onshift_partners = True, avg = delivery_partners_avgs[0], axis = 1)
    df.loc[df.total_busy_partners.isna(), "total_busy_partners"] = df.loc[df.total_busy_partners.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_busy_partners = True, avg = delivery_partners_avgs[1], axis = 1)
    df.loc[df.total_outstanding_orders.isna(), "total_outstanding_orders"] = df.loc[df.total_outstanding_orders.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_outstanding_orders = True, avg = delivery_partners_avgs[2], axis = 1)

    df.drop(["created_at", "actual_delivery_time", "actual_delivery_month", "created_at_month", "actual_delivery_month", "actual_deliveryt_weekday", "actual_delivery_hour"], axis = 1, inplace = True)

    return df

def final_preprocessing(train, test, outlier_cols, encoding_cols):

    features = train.drop("delivery_time", axis = 1).columns
    train, test = outliers_handler(train, test, outlier_cols)
    X_train, X_test = target_encoding(train, test, encoding_cols)
    X_train_sca, X_test_sca = standard_scaling(X_train, X_test)
    y_train, y_test = train.delivery_time.copy(), test.delivery_time.copy()

    return features, X_train_sca, X_test_sca, y_train, y_test