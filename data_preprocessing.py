import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split


def delivery_partners(df, given_info, total_onshift_partners = False, total_busy_partners = False, total_outstanding_orders = False):
    if (total_onshift_partners == True):
        total_onshift_partners = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.created_at_hour == df.created_at_hour)]["total_onshift_partners"]
        return total_onshift_partners.iloc[0]
    if (total_busy_partners == True):
        total_busy_partners = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.created_at_hour == df.created_at_hour)]["total_busy_partners"]
        return total_busy_partners.iloc[0]
    if (total_outstanding_orders == True):
        total_outstanding_orders = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.created_at_hour == df.created_at_hour)]["total_outstanding_orders"]
        return total_outstanding_orders.iloc[0]
    return "atleast choose one"

def order_protocol_value(df, given_info):
    order_protocol = given_info[(given_info.created_at_weekday == df.created_at_weekday) & (given_info.market_id == df.market_id)]["order_protocol"]
    return order_protocol.iloc[0]
    


df = pd.read_csv("datasets/dataset.csv")
print("--" * 90)
print(f"shape of intial data = {df.shape}")
print("--" * 90)

df.dropna(subset = ["actual_delivery_time"], inplace = True)

print(f"created_at and actual_delivery_time are used to get the time based features")

df["created_at"] = pd.to_datetime(df.created_at)
df["actual_delivery_time"] = pd.to_datetime(df.actual_delivery_time)
df["created_at_month"] = df.created_at.dt.month_name()
df["created_at_weekday"] = df.created_at.dt.day_name()
df["created_at_hour"] = df.created_at.dt.hour
df["actual_delivery_month"] = df.actual_delivery_time.dt.month_name()
df["actual_delivery_weekday"] = df.actual_delivery_time.dt.day_name()
df["actual_delivery_hour"] = df.actual_delivery_time.dt.hour
df["delivery_time"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds()/90

print(f"shape of before splitting data = {df.shape}")
print("--" * 90)

X = df.drop(["delivery_time"], axis = 1)
y = df["delivery_time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

print(f"the shape of train data : ")
print(f"   X_train = {X_train.shape}\n   y_train = {y_train.shape}")
print("--" * 50)
print(f"the shape of test data : ")
print(f"   X_test = {X_test.shape}\n   y_test = {y_test.shape}")
print("--" * 90)

print(f"filling NaN values in market_id\n{'--' * 50}")

most_market_id = X_train.market_id.value_counts().index[0]
market_id_map = X_train.groupby('store_id')['market_id'].apply(lambda x: x.mode()[0] if not x.isnull().all() else most_market_id).reset_index()

print(f"shape at begining (X_train) = {X_train.shape}")
market_nan = X_train[X_train.market_id.isna()].drop(["market_id"], axis = 1)
X_train.drop(X_train[X_train.market_id.isna()].index, axis = 0, inplace =  True)
print(f"shape after dropping market id NaN data = {X_train.shape}")
market_filled = market_nan.merge(market_id_map, on = ["store_id"], how = "left")
X_train = pd.concat([X_train, market_filled],axis = 0)
print(f"shape after concat = {X_train.shape}")
print(f"missing values in market id = {X_train.market_id.isna().sum()}")
print("--" * 50)

print(f"shape at begining (X_test) = {X_test.shape}")
market_nan = X_test[X_test.market_id.isna()].drop(["market_id"], axis = 1)
X_test.drop(X_test[X_test.market_id.isna()].index, axis = 0, inplace =  True)
print(f"shape after dropping market id NaN data = {X_test.shape}")
market_filled = market_nan.merge(market_id_map, on = ["store_id"], how = "left")
X_test = pd.concat([X_test, market_filled],axis = 0)
print(f"shape after concat = {X_test.shape}")
print(f"missing values in market id = {X_test.market_id.isna().sum()}")
X_test.market_id.fillna(most_market_id, inplace = True)
print(f"missing values in market id (after filling NaN with most frequent market) = {X_test.market_id.isna().sum()}")

print("--" * 90)

print(f"remaining missing valued features (train_data) = {X_train.columns[X_train.isna().sum() > 0].tolist()}")
print("--" * 50)
print(f"remaining missing valued features (test_data) = {X_test.columns[X_test.isna().sum() > 0].tolist()}")
print("--" * 90)

print(f"filling NaN values in total_onshift_partners, total_busy_partners, total_outstanding_orders \n{'--' * 50}")

train = pd.concat([X_train.reset_index(drop = True), y_train.reset_index(drop = True)], axis = 1)
delivery_partners_grps = train.groupby(by = ["created_at_weekday", "created_at_hour"])[["total_onshift_partners", "total_busy_partners", "total_outstanding_orders"]].apply("mean").round().reset_index()

X_train.loc[X_train.total_onshift_partners.isna(), "total_onshift_partners"] = X_train.loc[X_train.total_onshift_partners.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_onshift_partners = True, axis = 1)
X_train.loc[X_train.total_busy_partners.isna(), "total_busy_partners"] = X_train.loc[X_train.total_busy_partners.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_busy_partners = True, axis = 1)
X_train.loc[X_train.total_outstanding_orders.isna(), "total_outstanding_orders"] = X_train.loc[X_train.total_outstanding_orders.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_outstanding_orders = True, axis = 1)
print(f"missing values in X_train")
display(X_train.loc[:,["total_onshift_partners", "total_busy_partners", "total_outstanding_orders"]].isna().sum())
print("--" * 90)

X_test.loc[X_test.total_onshift_partners.isna(), "total_onshift_partners"] = X_test.loc[X_test.total_onshift_partners.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_onshift_partners = True, axis = 1)
X_test.loc[X_test.total_busy_partners.isna(), "total_busy_partners"] = X_test.loc[X_test.total_busy_partners.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_busy_partners = True, axis = 1)
X_test.loc[X_test.total_outstanding_orders.isna(), "total_outstanding_orders"] = X_test.loc[X_test.total_outstanding_orders.isna()].apply(delivery_partners, given_info = delivery_partners_grps, total_outstanding_orders = True, axis = 1)
print(f"missing values in X_test")
display(X_test.loc[:,["total_onshift_partners", "total_busy_partners", "total_outstanding_orders"]].isna().sum())
print("--" * 90)

print(f"remaining missing valued features (train_data) = {X_train.columns[X_train.isna().sum() > 0].tolist()}")
print("--" * 50)
print(f"remaining missing valued features (test_data) = {X_test.columns[X_test.isna().sum() > 0].tolist()}")
print("--" * 90)

print(f"filling NaN values in order_protocol\n{'--' * 50}")

order_protocol_grps = df.groupby(by = ["created_at_weekday", "market_id"])["order_protocol"].apply(lambda x : x.mode()[0]).reset_index()
X_train.loc[X_train.order_protocol.isna(), "order_protocol"] = X_train.loc[X_train.order_protocol.isna()].apply(order_protocol_value, given_info = order_protocol_grps, axis = 1)
X_test.loc[X_test.order_protocol.isna(), "order_protocol"] = X_test.loc[X_test.order_protocol.isna()].apply(order_protocol_value, given_info = order_protocol_grps, axis = 1)

print(f"remaining missing valued features (train_data) = {X_train.columns[X_train.isna().sum() > 0].tolist()}")
print("--" * 50)
print(f"remaining missing valued features (test_data) = {X_test.columns[X_test.isna().sum() > 0].tolist()}")
print("--" * 90)

X_train = X_train[X_train.subtotal > 0]
X_train = X_train[(X_train.total_busy_partners >= 0) & (X_train.total_onshift_partners >= 0) & (X_train.total_outstanding_orders >= 0)]
print(f"shape of the data (after dropping actual delivery time NaN vals, dropping subtotals <= 0, delivery partners < 0) = {X_train.shape}")
print("--" * 50)

X_test = X_test[X_test.subtotal > 0]
X_test = X_test[(X_test.total_busy_partners >= 0) & (X_test.total_onshift_partners >= 0) & (X_test.total_outstanding_orders >= 0)]
print(f"shape of the data (after dropping actual delivery time NaN vals, dropping subtotals <= 0, delivery partners < 0) = {X_test.shape}")
print("--" * 90)

X_train.drop(["actual_delivery_time", "created_at", "created_at_month", "actual_delivery_month", "actual_delivery_weekday", "actual_delivery_hour"], axis = 1, inplace = True)
print(X_train.columns)
print(X_train.shape)
print("--" * 50)

X_test.drop(["actual_delivery_time", "created_at", "created_at_month", "actual_delivery_month", "actual_delivery_weekday", "actual_delivery_hour"], axis = 1, inplace = True)
print(X_test.columns)
print(X_test.shape)
print("--" * 90)