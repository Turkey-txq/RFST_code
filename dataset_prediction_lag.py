'''
dataset prediction
'''


import pandas as pd
import xarray as xr
import numpy as np
import pickle



begin_lon = 89.4
end_lon = 103.51
begin_lat = 31
end_lat = 37.21


#==================================================
print('t_p_w_data')
t_p_w_data = xr.open_dataset(r'temp_prec_wind_month_CMFD_TRS.nc')
temp = t_p_w_data['temp']
prec = t_p_w_data['prec']
wind = t_p_w_data['wind']
print(temp.shape)
print(temp.time.values[0],temp.time.values[-1])

print('sce')
sce_avhrr = xr.open_dataset(r'avhrr_sce_month_TRS.nc')
sce = sce_avhrr['AVHRR_SCE']
print(sce.shape)
print(sce.time.values[0],sce.time.values[-1])

print('NDVI')
ndvi = xr.open_dataset(r'ndvi_month_TRS.nc')
ndvi = ndvi.transpose('time','lon','lat')
NDVI = ndvi['ndvi']
print(NDVI.shape)
print(NDVI.time.values[0],NDVI.time.values[-1])

print('DEM')
dem = xr.open_dataset(r'dem_data_TRS.nc')
DEM = dem['DEM']
print(DEM.shape)

print('soil_char')
soil_char = xr.open_dataset(r'soil_char_9depth_TRS.nc')
BD = soil_char['BD']
SA = soil_char['SA']
CL = soil_char['CL']
SI = soil_char['SI']
GRAV = soil_char['GRAV']
SOM = soil_char['SOM']
print(BD.shape)

# ============================================================
lat = dem['lat'].loc[begin_lat:end_lat]
lon = dem['lon'].loc[begin_lon:end_lon]
latin, lonin = np.meshgrid(lat,lon)
print(lonin.shape, latin.shape)
latin = np.array(latin).ravel()
lonin = np.array(lonin).ravel()
print(lonin.shape, latin.shape)

# ============================================================
varlist = ['LT80cm','LT160cm','LT320cm']
laglist = [1,2,3]
times = pd.date_range('1982-01', '2015-12',freq='MS')
for var,lag in zip(varlist,laglist):
    with open('regr_rf_month_'+var+'_13.pickle', 'rb') as f:
        regr = pickle.load(f)
    if var=='meantas':
        d = 0
    else:
        d = int(var[2:-2])
    print('var:',var)
    print('lag:',lag)
    BDin = np.array(BD.loc[d,:,:]).ravel()
    SAin = np.array(SA.loc[d,:,:]).ravel()
    CLin = np.array(CL.loc[d,:,:]).ravel()
    SIin = np.array(SI.loc[d,:,:]).ravel()
    GRAVin = np.array(GRAV.loc[d,:,:]).ravel()
    SOMin = np.array(SOM.loc[d,:,:]).ravel()
    DEMin = np.array(DEM.loc[:, :]).ravel()
    print(BDin.shape)

    for ti in times:
        print(ti)

        monthin = np.empty(shape=DEMin.shape)
        monthin[:] = ti.month
        print(monthin.shape)
        tin = np.array(temp.loc[ti, :, :]).ravel()- 273.15
        prein = np.array(prec.loc[ti, :, :]).ravel()*24
        windin = np.array(wind.loc[ti, :, :]).ravel()
        NDVIin = np.array(NDVI.loc[ti, :, :]).ravel()
        scein = np.array(sce.loc[ti, :, :]).ravel()*30
        print(tin.shape)

        xin = [DEMin, SAin, CLin, SIin, BDin, GRAVin, SOMin, scein, NDVIin,
              prein, tin, monthin, lonin, latin]  #
        x_predict = np.stack(xin, axis=-1)
        # 筛掉缺测值
        x_predict_ = x_predict[~np.isnan(x_predict).any(axis=1)]
        print(x_predict_[:,:-2].shape)
        y_predict = regr.predict(x_predict_[:,:-2])
        print(y_predict.shape)


        print('save data')
        data = np.empty(shape=DEM.shape)
        data[:] = np.nan
        lon = DEM.lon
        lat = DEM.lat
        data_output = xr.DataArray(data, coords=[lon, lat], dims=['lon', 'lat'])

        for i in range(len(y_predict)):
            # print(i)
            data_output.loc[x_predict_[i, -2],x_predict_[i, -1]] = y_predict[i]
        ti = ti + pd.Timedelta(days=(31*lag+1))
        data_output.to_netcdf(var+'_'+str(ti.year)+'_'+str(ti.month).zfill(2)+'_v13.nc')
        print('nc')
