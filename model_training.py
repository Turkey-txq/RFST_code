
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import time
import pickle

time0 = time.time()
#读取月平均数据
print('Read data1')
ta_ts_year = pd.read_excel('ta_ts_month.xlsx', header=0, names=['STATION','ALTITUDE','year', 'month',
                         'ta', 'meantas','LT5cm', 'LT10cm', 'LT15cm', 'LT20cm', 'LT40cm',
                         'LT80cm', 'LT160cm', 'LT320cm', 'prec', 'wind'])

meantas = ta_ts_year
#读取GST数据
varlist = ['meantas','LT5cm','LT10cm','LT15cm','LT20cm','LT40cm','LT80cm','LT160cm','LT320cm']#


#================================
print('Read data2')
ndvi_gimms = pd.read_excel('ndvi_gimms_month_TRS.xlsx',header=0,
                                names=['STA_NAME','STATION','LONGTITUDE', 'LATITUDE', 'ALTITUDE',
                                       'year', 'month','ndvi'])
ndvi = []
for i in range(len(meantas.STATION)):
    s = meantas.STATION.iloc[i]
    y = meantas.year.iloc[i]
    m = meantas.month.iloc[i]
    ndvi_data = ndvi_gimms.loc[(ndvi_gimms['STATION'] == s)&(ndvi_gimms['year']==y)
                               &(ndvi_gimms['month'] == m)]
    if ndvi_data.empty:
        ndvi.append(np.nan)
    else:
        ndvi.append(float(ndvi_data.ndvi))

#================================
print('Read data3')
sce = pd.read_excel('avhrr_sce_month_TRS.xlsx',header=0,
                                names=['STA_NAME','STATION','LONGTITUDE', 'LATITUDE', 'ALTITUDE',
                                       'year', 'month','sce'])
sce_avhrr = []
for i in range(len(meantas.STATION)):
    s = meantas.STATION.iloc[i]
    y = meantas.year.iloc[i]
    m = meantas.month.iloc[i]
    sce_data = sce.loc[(sce['STATION'] == s)&(sce['year']==y)
                               &(sce['month'] == m)]
    if sce_data.empty:
        sce_avhrr.append(np.nan)
    else:
        sce_avhrr.append(float(sce_data.sce))

#================================
print('Read data4')
soil_char_9depth = pd.read_excel('soil_char_9depth.xlsx',header=0,)

print(time.time()-time0,'s')


#================================choosing depth================================
varlist = ['meantas','LT5cm','LT10cm','LT15cm','LT20cm','LT40cm']


NDVIin = ndvi
scein1 = np.array(sce_avhrr)
DEMin = ta_ts_year['ALTITUDE']
tin = ta_ts_year['ta']
prein = ta_ts_year['prec']/10
windin = ta_ts_year['wind']

for var in varlist:
    time1 = time.time()
    soil_data2 = soil_char_9depth.loc[soil_char_9depth.depth == var]
    print(var)
    soil_char = pd.DataFrame()
    for i in meantas.STATION:
        data = soil_data2.loc[soil_data2['STATION'] == i]
        # soil_char = soil_char.append(data)
        soil_char = pd.concat([soil_char, data], ignore_index=True)

    BDin = np.array(soil_char.BD)
    SAin1 = np.array(soil_char.SA)
    CLin1 = np.array(soil_char.CL)
    SIin1 = np.array(soil_char.SI)
    GRAVin = np.array(soil_char.GRAV)
    SOMin = np.array(soil_char.SOM)

    meantasin = meantas[var]
    meantasin = np.array(meantasin)
    monthin = np.array(meantas.month)
    varsin = [DEMin, SAin1, CLin1, SIin1, BDin, GRAVin, SOMin, scein1, NDVIin,
              prein, tin, windin, monthin, meantasin]#

    # ===========================================================

    data_input = np.stack(varsin, axis=-1)
    print(data_input.shape)

    data_input = data_input[(~ta_ts_year['STATION'].isin([56046, 56021, 56029, 56067, 52957, 52974]))
               ]#& ta_ts_year['month'].isin([1, 2, 3, 11, 12])

    train_data_ = data_input[~np.isnan(data_input).any(axis=1)]
    y = train_data_[:, -1]
    x = train_data_[:, :-1]
    print(train_data_.shape)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    x_scaled = x
    y_scaled = y
    x_train_, x_test_, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.3, random_state=17)
    print('train:', x_train_.shape)
    print('test:', x_test_.shape)


    # =============================RF===============================
    print('RF')
    n_estimators = 60
    max_features = 0.5
    print(n_estimators, max_features)
    regr = ensemble.RandomForestRegressor(oob_score=True, n_estimators=n_estimators, max_features=max_features, random_state=6)# 调参,
    regr.fit(x_train_,y_train)
    y_predict = cross_val_predict(regr, x_train_, y_train, cv=5)


    from sklearn.metrics import mean_squared_error, mean_absolute_error
    model_mse = mean_squared_error(y_train, y_predict)  # 计算均方根误差
    model_mae = mean_absolute_error(y_train, y_predict)
    model_bias = y_train.mean() - y_predict.mean()
    # training
    r_rf0 = np.mean(np.multiply((y_train-np.mean(y_train)),(y_predict-np.mean(y_predict))))/(np.std(y_predict)*np.std(y_train))
    rmse_rf0 = np.sqrt(model_mse)
    bias_rf0 = model_bias
    r2_rf0 = r2_score(y_train,y_predict)
    print('R2:',r2_rf0)
    print('RMSE:', rmse_rf0)
    print('r:', r_rf0)
    print('bias:', bias_rf0)


    y_predict2 = regr.predict(x_test_)
    model_mse = mean_squared_error(y_test, y_predict2)  # 计算均方根误差
    model_mae = mean_absolute_error(y_test, y_predict2)
    model_bias = y_test.mean() - y_predict2.mean()
    # testing
    r_rf = np.mean(np.multiply((y_test-np.mean(y_test)),(y_predict2-np.mean(y_predict2))))/(np.std(y_predict2)*np.std(y_test))
    rmse_rf = np.sqrt(model_mse)
    bias_rf = model_bias
    r2_rf = r2_score(y_test,y_predict2)
    print('R2:',r2_rf)
    print('RMSE:', rmse_rf)
    print('r:', r_rf)
    print('bias:', bias_rf)


    print('save model')
    with open('regr_rf_month_'+var+'_13.pickle', 'wb') as f:
        pickle.dump(regr, f)
