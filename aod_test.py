# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 01:07:04 2023

@author: PRAMILA
"""
from __future__ import division
import streamlit as st
from pyhdf.SD import SD, SDC
import numpy as np
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import folium
import  io, sys, json,glob
from urllib.request import urlopen
from urllib.request import urlopen
import urllib.request
import urllib.request, json
 # ensures no rounding errors from division involving integers
from math import * # enables use of pi, trig functions, and more.
import smtplib
import pandas as pd # gives us the dataframe concept
pd.options.display.max_columns = 50
pd.options.display.max_rows = 9
import streamlit as st
from pyhdf.SD import SD, SDC

#from mpl_toolkits.basemap import Basemap

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import h5py
import datetime
opener = urllib.request.build_opener()
opener.addheaders = [('Authorization', 'Bearer cHJhbWlsYV9tYW5pY2thdmFzYWthbjpjSEpoYldsc1lTNHhPVEF4TVRNM1FITnlaV011WVdNdWFXND06MTY2MTQ1NDk2NjoxMTNkMjk3NjJjYWVmMjk0ZWRkODEwZTgwMjM2YmJkNzRlM2ExMDhh')]
urllib.request.install_opener(opener)
st.markdown("<h1 style ='color:green; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>Impact of Aerosols in Solar Power Generation </h1>", unsafe_allow_html=True) 


st.markdown("<h1 style='text-align: left; font-weight:bold;color:black;background-color:white;font-size:11pt;'> Choose any Location </h1>",unsafe_allow_html=True)

m = folium.Map()
m.add_child(folium.LatLngPopup())
map = st_folium(m, height=350, width=700)
try:

	user_lat=map['last_clicked']['lat']
	user_lon=map['last_clicked']['lng'] 

	st.write(user_lat)
	st.write(user_lon)
except:
	st.warning("No location choosen")
plantsize_type= st.radio("Choose any one of Solar Panel Capacity you want to install",('Individual home/flat','Residential Flat','Commerical Industry'))
if(plantsize_type=="Individual home/flat"):
    plantsize=1
elif(plantsize_type=="Residential Flat"):
    plantsize=5    
else:
    plantsize=50  
area=st.number_input('Please enter the area of panel(in m2) :')    
today = datetime.date.today()-datetime.timedelta(days=1)
mail=st.text_input('Please enter your mail id: ')
date = st.date_input('📅 Date', value = today,max_value=today)
date=date.strftime("%Y/%m/%d")
fmt = '%Y/%m/%d'
dt = datetime.datetime.strptime(date, fmt)
tt = dt.timetuple()
julian_day=tt.tm_yday
if st.button("Predict"):
    user_lat=float(user_lat)
    user_lon=float(user_lon)
    filePath=r"MOD08_M3.A2022335.061.2023003231743.hdf"
    hdf= SD(filePath, SDC.READ)
    #print(hdf.datasets())
    sds=hdf.select('AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean')
    data=sds.get()
    #print(hdf.datasets())
    lat = hdf.select('XDim')
    latitude = lat[:]
    print(latitude)
    
    min_lat=latitude.min()
    max_lat=latitude.max()
 
    
    lon = hdf.select('YDim')
    longitude = lon[:]
    min_lon=longitude.min()
    max_lon=longitude.max()
    	   
    attributes=sds.attributes()
    scale_factor=attributes['scale_factor']
    print("scale_factor  ",scale_factor)
    fillvalue=attributes['_FillValue']

    data_value=data[int(user_lat),int(user_lat)]
    if (data_value==fillvalue):
         	print('The value of AOD at this pixel is',fillvalue,',(No Value)\n')
         	AOD500nm=0
    else:
         	print('The value of AOD at this pixel is ',round(data_value*scale_factor,3))
         	AOD500nm=round(data_value*scale_factor,3)
    #st.write(AOD500nm)
    st.success("AOD Value {} ".format(round(AOD500nm,3)))
    phi = user_lat
    longitude = user_lon
    tz = -7
    P_mb = 840
    Ozone_cm = 0.3
    H20_cm = 1.5
    
    AOD380nm = 0.15
    Taua = (0.2758*AOD380nm)+(0.35*AOD500nm)
    Ba = 0.85
    albedo = 0.2
    
    G_sc = 1367 # W/m^2
    std_mer = longitude-longitude%15+15 # This Standard Meridian calculation is only a guide!! 
                                        # Please double check this value for your location!
    
    ### Day of the Year Column
    
    n = julian_day # julian day of the year
    n_hrly = list(pd.Series(n).repeat(24)) # julian day numbers repeated hourly to create 8760 datapoints in dataset
    
    ds = pd.DataFrame(n_hrly, columns=['DOY']) # create dataframe with julian days 
    
    ### Hr of the Day Column
    
    ds['HR'] = [(hr)%24 for hr in ds.index.tolist()] # append dataframe with hr of the day for each day
    
    ### Extraterrestrial Radiation
    
    def etr(n):
        return G_sc*(1.00011+0.034221*cos(2*pi*(n-1)/365)+0.00128*sin(2*pi*(n-1)/365)+0.000719*cos(2*(2*pi*(n-1)/365))+0.000077*sin(2*(2*pi*(n-1)/365)))
    
    ds['ETR'] = [etr(n) for n in ds['DOY']] # append dataframe with etr for day
    
    ### Intermediate Parameters
    
    ds['Dangle'] = [2*pi*(n-1)/365 for n in ds['DOY']]

    def decl(Dangle):
        return (0.006918-0.399912*cos(Dangle)+0.070257*sin(Dangle)-0.006758*cos(2*Dangle)+0.000907*sin(2*Dangle)-0.002697*cos(3*Dangle)+0.00148*sin(3*Dangle))*(180/pi)
    ds['DEC'] = [decl(Dangle) for Dangle in ds['Dangle']]
    
    def eqtime(Dangle):
        return (0.0000075+0.001868*cos(Dangle)-0.032077*sin(Dangle)-0.014615*cos(2*Dangle)-0.040849*sin(2*Dangle))*229.18
    ds['EQT'] = [eqtime(Dangle) for Dangle in ds['Dangle']]
    
    def omega(hr, eqt):
        return 15*(hr-12.5) + longitude - tz*15 + eqt/4
    ds['Hour Angle'] = [omega(hr, eqt) for hr, eqt in zip(ds['HR'],ds['EQT'])]
    
    def zen(dec, hr_ang):
        return acos(cos(dec/(180/pi))*cos(phi/(180/pi))*cos(hr_ang/(180/pi))+sin(dec/(180/pi))*sin(phi/(180/pi)))*(180/pi)
    ds['Zenith Ang'] = [zen(dec, hr_ang) for dec, hr_ang in zip(ds['DEC'],ds['Hour Angle'])]
    
    def airmass(zenang):
        if zenang < 89:
            return 1/(cos(zenang/(180/pi))+0.15/(93.885-zenang)**1.25)
        else:
            return 0
    ds['Air Mass'] = [airmass(zenang) for zenang in ds['Zenith Ang']]
    
    ### Intermediate Results
    
    def T_rayleigh(airmass):
        if airmass > 0:
            return exp(-0.0903*(P_mb*airmass/1013)**0.84*(1+P_mb*airmass/1013-(P_mb*airmass/1013)**1.01))
        else:
            return 0
    ds['T rayleigh'] = [T_rayleigh(airmass) for airmass in ds['Air Mass']]
    
    def T_ozone(airmass):
        if airmass > 0:
            return 1-0.1611*(Ozone_cm*airmass)*(1+139.48*(Ozone_cm*airmass))**-0.3034-0.002715*(Ozone_cm*airmass)/(1+0.044*(Ozone_cm*airmass)+0.0003*(Ozone_cm*airmass)**2)
        else:
            return 0
    ds['T ozone'] = [T_ozone(airmass) for airmass in ds['Air Mass']]
    
    def T_gasses(airmass):
        if airmass > 0:
            return exp(-0.0127*(airmass*P_mb/1013)**0.26)
        else:
            return 0
    ds['T gases'] = [T_gasses(airmass) for airmass in ds['Air Mass']]
    
    def T_water(airmass):
        if airmass > 0:
            return 1-2.4959*airmass*H20_cm/((1+79.034*H20_cm*airmass)**0.6828+6.385*H20_cm*airmass)
        else:
            return 0
    ds['T water'] = [T_water(airmass) for airmass in ds['Air Mass']]
    
    def T_aerosol(airmass):
        if airmass > 0:
            return exp(-(Taua**0.873)*(1+Taua-Taua**0.7088)*airmass**0.9108)
        else:
            return 0
    ds['T aerosol'] = [T_aerosol(airmass) for airmass in ds['Air Mass']]
    
    def taa(airmass, taerosol):
        if airmass > 0:
            return 1-0.1*(1-airmass+airmass**1.06)*(1-taerosol)
        else:
            return 0
    ds['TAA'] = [taa(airmass, taerosol) for airmass, taerosol in zip(ds['Air Mass'],ds['T aerosol'])]
    
    def rs(airmass, taerosol, taa):
        if airmass > 0:
            return 0.0685+(1-Ba)*(1-taerosol/taa)
        else:
            return 0
    ds['rs'] = [rs(airmass, taerosol, taa) for airmass, taerosol, taa in zip(ds['Air Mass'],ds['T aerosol'],ds['TAA'])]
    
    def Id(airmass, etr, taerosol, twater, tgases, tozone, trayleigh):
        if airmass > 0:
            return 0.9662*etr*taerosol*twater*tgases*tozone*trayleigh
        else:
            return 0
    ds['Id'] = [Id(airmass, etr, taerosol, twater, tgases, tozone, trayleigh) for airmass, etr, taerosol, twater, tgases, tozone, trayleigh in zip(ds['Air Mass'],ds['ETR'],ds['T aerosol'],ds['T water'],ds['T gases'],ds['T ozone'],ds['T rayleigh'])]
    
    def idnh(zenang, Id):
        if zenang < 90:
            return Id*cos(zenang/(180/pi))
        else:
            return 0
    ds['IdnH'] = [idnh(zenang, Id) for zenang, Id in zip(ds['Zenith Ang'],ds['Id'])]
    
    def ias(airmass, etr, zenang, tozone, tgases, twater, taa, trayleigh, taerosol):
        if airmass > 0:
            return etr*cos(zenang/(180/pi))*0.79*tozone*tgases*twater*taa*(0.5*(1-trayleigh)+Ba*(1-(taerosol/taa)))/(1-airmass+(airmass)**1.02)
        else:
            return 0
    ds['Ias'] = [ias(airmass, etr, zenang, tozone, tgases, twater, taa, trayleigh, taerosol) for airmass, etr, zenang, tozone, tgases, twater, taa, trayleigh, taerosol in zip(ds['Air Mass'],ds['ETR'],ds['Zenith Ang'],ds['T ozone'],ds['T gases'],ds['T water'],ds['TAA'],ds['T rayleigh'],ds['T aerosol'])]
    
    def gh(airmass, idnh, ias, rs):
        if airmass > 0:
            return (idnh+ias)/(1-albedo*rs)
        else:
            return 0
    ds['GH'] = [gh(airmass, idnh, ias, rs) for airmass, idnh, ias, rs in zip(ds['Air Mass'],ds['IdnH'],ds['Ias'],ds['rs'])]
    
    ### Decimal Time
    
    def dectime(doy, hr):
        return doy+(hr-0.5)/24
    ds['Decimal Time'] = [dectime(doy, hr) for doy, hr in zip(ds['DOY'],ds['HR'])]
    
    ### Model Results (W/m^2)
    
    ds['Direct Beam'] = ds['Id']
    
    ds['Direct Hz'] = ds['IdnH']
    
    ds['Global Hz'] = ds['GH']
    
    ds['Dif Hz'] = ds['Global Hz']-ds['Direct Hz']
    
    #ds[11:15]
    
    #st.write(ds)
    pylab.rcParams['figure.figsize'] = 16, 6  # this sets the default image size for this session
    
    ax = ds[ds['DOY']==212].plot('HR',['Global Hz','Direct Hz','Dif Hz'],title='Bird Clear Sky Model Results')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Irradiance W/m^2")
    majorx = ax.set_xticks(range(0,25,1))
    majory = ax.set_yticks(range(0,1001,200))
    
    #min_index=ds['Direct Beam'].argmin()
    #print(min_index)
    #st.write(ds['Direct Beam'].mean())
    diffuse_solar_irradiance=(ds['Dif Hz'].mean())+(ds['Direct Hz'].mean())
    st.info("Avg Solar Irradiance per day {} W/m2".format(round(diffuse_solar_irradiance,2)))
    solar_irradiance=diffuse_solar_irradiance
    
    #plantsize=2#46kw
    total_electricity=solar_irradiance*0.0036*1.1*plantsize*area
    #savings
    electricity_per_unit=8#8/kwh
    
    monthly_savings=(total_electricity/12)*electricity_per_unit
    annual_savings=(total_electricity)*electricity_per_unit
    

    st.success("Total electricity from solar plant(per day`5)+` {} kW".format(round(total_electricity,2)))
    st.write("1 Unit Cost:₹8")
    st.info("FINANCIAL SAVINGS")
    st.markdown(f"""<h1 style='text-align: left; font-weight:bold;color:black;background-color:yellow;font-size:11pt;'>Monthly savings: ₹ <mark style="background-color:white">{format(round(monthly_savings,2))} </mark> </h1>""",unsafe_allow_html=True)
    #st.warning("Electricity Cost per Day: ₹ {}".format(round((solar_irradiance*9*24)/1000,2)))
    st.markdown(f"""<h1 style='text-align: left; font-weight:bold;color:black;background-color:pink;font-size:11pt;'>Annual savings: ₹ <mark style="background-color:white">{format(round(annual_savings,2))}</mark> </h1>""",unsafe_allow_html=True)
    #st.info("Electricity Cost per Month: ₹ {}".format(round((solar_irradiance*9*720)/1000,2)))
    final_loss = total_electricity*0.03
    final_loss_per = (final_loss/total_electricity)*100
    st.error("The loss percentage is:{} ".format(final_loss_per))
    st.balloons()



           
           	           
           
           
      
    
