from math import radians, cos, sin, asin, sqrt
import math
import pandas as pd
import numpy as np
import csv
import os 
import metpy.calc
from metpy.units import units


def haversine(lon1, lat1, lon2, lat2, timestep):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371*1000 # Radius of earth in kilometers. Use 3956 for miles
    wspeed = c * r/timestep
    return wspeed


def getDir(lon1, lat1, lon2, lat2):
    lata = math.radians(lat1)
    lona = math.radians(lon1)
    latb = math.radians(lat2)
    lonb = math.radians(lon2)
    dLon = math.radians(lon2 - lon1)

    x = math.sin(dLon) * math.cos(latb)
    y = math.cos(lata) * math.sin(latb) - (math.sin(lata) * math.cos(latb) * math.cos(dLon))
    θ = math.atan2(x, y)
    initial_bearing = (θ*180/math.pi + 360) % 360
    wdir = initial_bearing - 180
    if wdir < 0:
        wdir  = wdir + 360
    return wdir

############################################
#                   main
############################################
filepath = '/Users/zerojerry/Desktop/NCAR_Meet_Summary/wind_data/'
outpath  = '/Users/zerojerry/Desktop/NCAR_Meet_Summary/wind_test/'
all_files = os.listdir("/Users/zerojerry/Desktop/NCAR_Meet_Summary/wind_data/")
outfile = [sub[ : -4] for sub in all_files] 

list_of_values = list(range(0, 9000, 120))

for i in range(0, len(outfile)):
    df = pd.read_csv(filepath + all_files[i], skiprows = 1, header = 0, sep = ',')
    df.columns = df.columns.str.strip()
    df.replace({99999: np.nan}, inplace=True)
    df = df[(df['Elapsed time'] >= 0) | (df['Elapsed time'].isnull())]
    sel_data = df[['Elapsed time','Latitude','Longitude','GeopotHeight',]].dropna()
    sel_data = sel_data.rename({'Elapsed time': 'Elapsed_time'}, axis=1)  # new method

    geophgt = sel_data['GeopotHeight'].tolist()
    height = metpy.calc.geopotential_to_height(geophgt * units('m^2/s^2')) 

    #sel_data['GeopotHeight'] = height
    sel_data = sel_data.rename({'GeopotHeight': 'Height'}, axis=1)  # new method
    final = sel_data[sel_data.Elapsed_time.isin(list_of_values)]
    count_row = final.shape[0]
    final['Latitude'] = round(final['Latitude'],4)
    final['Longitude'] = round(final['Longitude'],4)
    final['Height'] = round(final['Height'],4)
    final['Elapsed_time'] = final['Elapsed_time']- 60.
    windspeed = [np.nan]
    winddir   = [np.nan]
    print(all_files[i])
    for n in range(0, count_row-1):
        lat1 = final.at[final.index[n], 'Latitude']
        lon1 = final.at[final.index[n], 'Longitude']
        lat2 = final.at[final.index[n+1], 'Latitude']
        lon2 = final.at[final.index[n+1], 'Longitude']
        time = 120.
        windspeed.append(round(haversine(lon1, lat1, lon2, lat2, time), 3))
        winddir.append(round(getDir(lon1, lat1, lon2, lat2), 3))

    final['wind_speed'] = windspeed
    final['wind_direction'] = winddir
    final.replace({np.nan:0.}, inplace=True)
    final['Height'][0] = 0
    final = final.iloc[1:]
    final.to_csv(outpath + outfile[i]+'.csv',index=False,na_rep=np.nan)

    #df = pd.DataFrame(None)
    #sel_data = pd.DataFrame(None)
    #final = pd.DataFrame(None)
    #windspeed.clear()
    #winddir.clear()
    #height.clear()
    #geophgt.clear()