import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
###
#train
#SK_ID_CURR - PK
#TARGET - the label we want to predict
#bureau
#SK_ID_BUREAU - PK
#SK_ID_CURR - FK to train
#b_balance
#SK_ID_BUREAU - FK to bureau
#pre_app
#SK_ID_PREV - PK
#SK_ID_CURR - FK to train

train = pd.read_csv("..\\Data\\application_train.csv")
bureau = pd.read_csv("..\\Data\\bureau.csv")
b_balance = pd.read_csv("..\\Data\\bureau_balance.csv")
prev_app = pd.read_csv('..\\Data\\previous_application.csv')
cc_balance = pd.read_csv("..\\Data\\credit_card_balance.csv")
pos_cash_balance = pd.read_csv("..\\Data\\POS_CASH_balance.csv")
installments = pd.read_csv("..\\Data\\installments_payments.csv")

def credit_active_count(data):
    active_count = sum(data['CREDIT_ACTIVE'] == "Active")
    closed_count = sum(data['CREDIT_ACTIVE'] == "Closed")
    sold_count = sum(data['CREDIT_ACTIVE'] == "Sold")
    bad_count = sum(data['CREDIT_ACTIVE'] == "Bad")
    all_count = pd.Series([active_count, closed_count, sold_count, bad_count], 
                          index = ['active_count', 'closed_count', 'sold_count', 'bad_count'])
    return all_count

credit_active_count_df = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].groupby("SK_ID_CURR").apply(credit_active_count)
bureau_agg_stat = bureau.groupby("SK_ID_CURR").aggregate({'DAYS_CREDIT': ['mean'], 
                                                          'CREDIT_DAY_OVERDUE': ['mean'], 
                                                          'AMT_CREDIT_MAX_OVERDUE': ['max'],
                                                          'CNT_CREDIT_PROLONG': ['count', 'sum', 'mean'],
                                                          'AMT_CREDIT_SUM': ['min', 'max', 'mean']})
bureau_agg_stat.columns = ['mean_days_btw_application', 'mean_overdue_day', 
                           'max_overdue_amt', 'cnt_credit', 'sum_prolong_cnt', 
                           'mean_prolong_cnt', 'min_credit', 'max_credit', 'mean_credit']
bureau_agg_stat = pd.merge(credit_active_count_df, bureau_agg_stat, left_index=True, right_index=True)


train_master = pd.merge(train, bureau_agg_stat, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
train_master.dropna(axis = 0, how ="any", inplace = True)
X = train_master[['mean_days_btw_application', 'mean_overdue_day', 
                  'max_overdue_amt', 'cnt_credit', 'sum_prolong_cnt', 
                  'mean_prolong_cnt', 'min_credit', 'max_credit', 
                  'mean_credit', 'active_count', 'closed_count', 
                  'sold_count', 'bad_count']]
y = train_master['TARGET']



from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#classifier = SVC(class_weight = 'balanced')
classifier = LogisticRegression(class_weight = 'balanced')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
sum(y_pred == y_test)/len(y_test)
#sum(y_pred)
#y_pred_adjusted = [1 if item > 0.1 else 0 for item in classifier.decision_function(X_test)]
#sum(y_pred_adjusted == y_test)/len(y_test)
