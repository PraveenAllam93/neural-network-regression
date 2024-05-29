import numpy as np
import pandas as pd
from scipy.stats import shapiro

def basic_info(df):

    """
    Takes a DataFrame as input, and gives the basic info like shape, missing values count, duplicated rows, unique values and dtypes of the features
    
    Args:
        df (pandas.DataFrame): The DataFrame for which you want the details of

    Returns: 
        None 
    """
    print(f"shape of the date : \n\trows = {df.shape[0]}, columns = {df.shape[1]}\n")
    missing_val_count = df.isna().sum().sum()
    print(f"missing values: \n\tcount = {missing_val_count}")
    if missing_val_count != 0:
        missing_data = df.isna().sum().reset_index().rename({"index" : "feature", 0 : "missing_val_count"}, axis = 1)
        missing_data["missing_val_percentage"] = np.round((missing_data["missing_val_count"] / df.shape[0]) * 100, 2)
        missing_data = missing_data.sort_values(by = "missing_val_count", ascending = False)
        display(missing_data)

    print(f"duplicated records: \n\tcount = {df.duplicated().sum()}\n")
    print(f"Unique Values : ")
    nunique_vals = df.nunique().reset_index().rename({"index" : "feature", 0 : "nunique_vals"}, axis = 1)
    display(nunique_vals)

    display(df.dtypes.reset_index().rename(columns = {"index" : "fetaure", 0 : "data type"}))

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

def order_protocol_value(df, given_info):
    order_protocol = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.market_id == df.market_id)]["order_protocol"]
    return order_protocol.iloc[0]

def store_category(df, given_info, most_store_primary_category):
    required_data = given_info[(given_info.created_at_hour == df.created_at_hour) & (given_info.created_at_weekday == df.created_at_weekday) & (given_info.market_id == df.market_id) & (given_info.store_id == df.store_id)][["store_primary_category", "subtotal"]]
    if required_data.shape[0] == 0:
        required_data = given_info[(given_info.created_at_hour == df.created_at_hour) & (given_info.created_at_weekday == df.created_at_weekday) & (given_info.market_id == df.market_id)][["store_primary_category", "subtotal"]]
    if required_data.shape[0] == 0:
        required_data = given_info[(given_info.created_at_hour == df.created_at_hour) & (given_info.created_at_weekday == df.created_at_weekday)][["store_primary_category", "subtotal"]]
    if required_data.shape[0] == 0:
        required_data = given_info[(given_info.created_at_hour == df.created_at_hour)][["store_primary_category", "subtotal"]]
    if required_data.shape[0] == 0:
        return most_store_primary_category    
    
    required_data["subtotal_diff"] = abs(required_data["subtotal"] - df.subtotal)
    required_data.sort_values(by = ["subtotal_diff"], ascending = True, inplace = True)
    return required_data.iloc[0,0]

def market_map(df, given_info, most_market_id):
    required_data = given_info[(given_info.store_id == df.store_id)]["market_id"]
    if required_data.shape[0] > 0:
        return required_data.iloc[0]
    return most_market_id    


def shapiro_test(df):

    stat, p = shapiro(df)
    alpha = 0.05

    # print("Shapiro-Wilk Test Statistic:", stat)
    # print("p-value:", p)
    if p > alpha:
        print("Sample looks Gaussian (fail to reject H0)")
    else:
        print("Sample does not look Gaussian (reject H0)")
