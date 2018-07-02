# !/usr/bin/python

'''
This script is to clean data collected over trials conducted in Cambridge Bay, NU, Canada in June/July 2018. This script prepares to clean text and CSV scripts from CTD, GGA, Nirtrate, Optode, Airmar, Pixhawk, and two TDGP -- one prototype, and one Pro-Oceanus.

Maintainer: Victoria Preston
Supervisor: Anna Michel
Contact: vpreston@whoi.edu, vpreston@mit.edu
'''

import sensor_cleaning as sc
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import gpxpy


def clean_rbr_ctd(filepath, bounds=None, plot_timeseries=False):
	'''
	Clean textfiles coming from an RBR Concerto CTD, as generated by the RBR software.
	Inputs:
		filepath (string): path to the file from which to copy
		bounds (list of floats): julian dates on which to truncate the data
		plot_timeseries (bool): whether to plot a timeseries of the cleaned CTD data
	'''
	ctd_df = pd.read_table(filepath[0], delimiter=',', header=0)
	for m in filepath[1:]:
		temp = pd.read_table(m, delimiter=',', header=0)
		ctd_df = ctd_df.append(temp, ignore_index=True)
	ctd_df = sc.clean_ctd(ctd_df)

	#if bounds are present, only grab those bounds
	if bounds is not None:
		ctd_df = ctd_df.drop(ctd_df[ctd_df.Julian_Date <= (bounds[0])].index)
		ctd_df = ctd_df.drop(ctd_df[ctd_df.Julian_Date >= (bounds[1])].index)

	ctd_df = ctd_df.dropna()

	if plot_timeseries == True:
		plt.plot(ctd_df['Julian_Date'], ctd_df['Salinity'])
		print ctd_df.head(2)

	return ctd_df

def clean_rpi_ctd(filepath, bounds=None, plot_timeseries=False):
	'''
	Clean textfiles logged by the raspberry pi of the CTD data (through RF transmitter).
	Inputs:
		filepath (string): path to the file from which to copy
		bounds (list of floats): Julian dates on which to truncate data
		plot_timeseries (bool): whether to plot a timeseries of the cleaned CTD data
	'''
	ctd_df = pd.read_table(filepath[0], delimiter=',', header=1, engine='c')
	ctd_df.columns = ['Misc', 'Time', 'Conductivity', 'Temperature', 'Pressure', 'Sea pressure', 'Depth', 'Salinity', 'Speed of sound', 'Specific conductivity']
	for m in filepath[1:]:
		temp = pd.read_table(m, delimiter=',', header=1, engine='c')
		temp.columns = ['Misc', 'Time', 'Conductivity', 'Temperature', 'Pressure', 'Sea pressure', 'Depth', 'Salinity', 'Speed of sound', 'Specific conductivity']
		ctd_df = ctd_df.append(temp, ignore_index=True)
	ctd_df = sc.clean_ctd(ctd_df)

	#if bounds are present, only grab those bounds
	if bounds is not None:
		ctd_df = ctd_df.drop(ctd_df[ctd_df.Julian_Date <= (bounds[0])].index)
		ctd_df = ctd_df.drop(ctd_df[ctd_df.Julian_Date >= (bounds[1])].index)

	ctd_df = ctd_df.dropna()

	if plot_timeseries == True:
		plt.plot(ctd_df['Julian_Date'], ctd_df['Salinity'])
		print ctd_df.head(2)
	
	return ctd_df

def clean_rpi_nitrate(filepath, bounds=None, plot_timeseries=False, day=29, month=6, year=2018):
	'''
	Clean textfiles logged by the raspberry pi of the nitrate data.
	Inputs:
		filepath (string): path to the file from which to copy
		bounds (list of floats): Julian dates on which to truncate data
		plot_timeseries (bool): whether to plot a timeseries of the cleaned CTD data
	'''
	nit_df = pd.read_table(filepath[0], delimiter=',', header=1, engine='c')
	nit_df = nit_df.drop(nit_df.columns[5:], axis=1)
	nit_df.columns = ['misc', 'sats', 'y', 'time', 'data']
	for m in filepath[1:]:
		temp = pd.read_table(m, delimiter=',', header=1, engine='c')
		temp = temp.drop(temp.columns[5:], axis=1)
		temp.columns = ['misc', 'sats', 'y', 'time', 'data']
		nit_df = nit_df.append(temp, ignore_index=True)
	nit_df = sc.clean_nitrate(nit_df, day, month, year)

	# # #keep only certain columns
	nit_df = nit_df.loc[:,['Julian_Date',
                         'data',
                         'Year',
                         'Month',
                         'Day',
                         'Hour',
                         'Minute',
                         'Second',
                         'Seconds_Elapsed']]

	#if bounds are present, only grab those bounds
	if bounds is not None:
		nit_df = nit_df.drop(nit_df[nit_df.Julian_Date <= (bounds[0])].index)
		nit_df = nit_df.drop(nit_df[nit_df.Julian_Date >= (bounds[1])].index)

	nit_df = nit_df.dropna()

	if plot_timeseries == True:
		plt.plot(nit_df['Julian_Date'], nit_df['data'])
		print nit_df.head(2)

	return nit_df

def clean_rpi_optode(filepath, offset=2440678.484, bounds=None, plot_timeseries=False):
	'''
	Clean textfiles logged by the raspberry pi of the optode data.
	Inputs:
		filepath (string): path to the file from which to copy
		bounds (list of floats): Julian dates on which to truncate data
		plot_timeseries (bool): whether to plot a timeseries of the cleaned CTD data
	'''
	op_df = pd.read_table(filepath[0], delimiter=',', header=0, engine='c')
	for m in filepath[1:]:
		temp = pd.read_table(m, delimiter=',', header=0, engine='c')
		op_df = op_df.append(temp, ignore_index=True)
	op_df = sc.clean_optode(op_df, offset)

	# #keep only certain columns
	

	#if bounds are present, only grab those bounds
	if bounds is not None:
		op_df = op_df.drop(op_df[op_df.Julian_Date <= (bounds[0])].index)
		op_df = op_df.drop(op_df[op_df.Julian_Date >= (bounds[1])].index)

	op_df = op_df.dropna()

	if plot_timeseries == True:
		plt.plot(op_df['Julian_Date'], op_df['O2Concentration']*0.2)
		print op_df.head(2)
	
	return op_df

def clean_airmar(filepath, bounds=None, plot_timeseries=False, offset=0.0):
	'''
	Clean textfiles logged by the raspberry pi of the airmar data.
	Inputs:
		filepath (string): path to the file from which to copy
		bounds (list of floats): Julian dates on which to truncate data
		plot_timeseries (bool): whether to plot a timeseries of the cleaned CTD data
	'''
	air_df = pd.read_table(filepath[0], delimiter=',', header=0, engine='c')
	air_df.columns = ['lat', 'lon', 'alt_M', 'geo_sep_M', 'COG_T', 'SOG_K', 'TOD', 'day', 'month', 'year', 'pressure_B', 'air_temp_C', 'wind_dir_T', 'wind_speed_M', 'reference', 'rateofturn', 'rel_wind_chill_c', 'theo_wind_chill_c', 'misc']
	for m in filepath[1:]:
		temp = pd.read_table(m, delimiter=',', header=0, engine='c')
		temp.columns = ['lat', 'lon', 'alt_M', 'geo_sep_M', 'COG_T', 'SOG_K', 'TOD', 'day', 'month', 'year', 'pressure_B', 'air_temp_C', 'wind_dir_T', 'wind_speed_M', 'reference', 'rateofturn', 'rel_wind_chill_c', 'theo_wind_chill_c', 'misc']
		air_df = air_df.append(temp, ignore_index=True)

	air_df = sc.clean_airmar(air_df)
	air_df.loc[:,'Julian_Date'] = air_df.apply(lambda x: x['Julian_Date']-offset,axis=1)

	#if bounds are present, only grab those bounds
	if bounds is not None:
		air_df = air_df.drop(air_df[air_df.Julian_Date <= (bounds[0])].index)
		air_df = air_df.drop(air_df[air_df.Julian_Date >= (bounds[1])].index)

	air_df = air_df.dropna()
	air_df = air_df.drop(air_df[air_df.lat == 0.].index)

	if plot_timeseries == True:
		plt.plot(air_df['lat_mod'], air_df['lon_mod'],'g*')
		print air_df.head(2)
	
	return air_df

def clean_gga(filepath, offset_time = -0.003, bounds=None, plot_timeseries=False):
	'''
	Clean textfiles logged by the LGR GGA.
	Inputs:
		filepath (string): path to the file from which to copy
		bounds (list of floats): Julian dates on which to truncate data
		plot_timeseries (bool): whether to plot a timeseries of the cleaned CTD data
	'''
	gga_df = pd.read_table(filepath[0], delimiter=',', header=1, engine='c')
	gga_df.columns = [m.strip(' ') for m in gga_df.columns]
	for m in filepath[1:]:
		temp = pd.read_table(m, delimiter=',', header=1, engine='c')
		temp.columns = [l.strp(' ') for l in temp.columns]
		gga_df = gga_df.append(temp, ignore_index=True)

	gga_df = sc.clean_gga(gga_df, offset_time)
	
	#if bounds are present, only grab those bounds
	if bounds is not None:
		gga_df = gga_df.drop(gga_df[gga_df.Julian_Date <= (bounds[0])].index)
		gga_df = gga_df.drop(gga_df[gga_df.Julian_Date >= (bounds[1])].index)

	gga_df = gga_df.dropna()

	if plot_timeseries == True:
		plt.plot(gga_df['Julian_Date_cor'], gga_df['CH4_ppm_adjusted'])
		print gga_df.head(2)
	
	return gga_df

def clean_pix(filepath, bounds=None, plot_timeseries=False):
	'''
	Clean the gpx data from the pixhawk for the trials
	'''	
	pix_df = gpxpy.parse(open(filepath[0]))
	track = pix_df.tracks[0]
	segment = track.segments[0]

	data = []
	segment_length = segment.length_3d()
	for i, point in enumerate(segment.points):
		data.append([point.longitude, point.latitude, point.elevation, point.time, segment.get_speed(i)])

	p_df = pd.DataFrame(data, columns=['Longitude', 'Latitude', 'Altitude', 'Time', 'Speed'])
	p_df.loc[:,'Year'] = p_df.apply(lambda x : int(x['Time'].year),axis=1)
	p_df.loc[:,'Month'] = p_df.apply(lambda x : int(x['Time'].month),axis=1)
	p_df.loc[:,'Day'] = p_df.apply(lambda x : int(x['Time'].day),axis=1)
	p_df.loc[:,'Hour'] = p_df.apply(lambda x : float(x['Time'].hour+4.),axis=1)
	p_df.loc[:,'Minute'] = p_df.apply(lambda x : float(x['Time'].minute),axis=1)
	p_df.loc[:,'Second'] = p_df.apply(lambda x : float(x['Time'].second),axis=1)

	p_df = sc.global_time_column(p_df)

	if bounds is not None:
		p_df = p_df.drop(p_df[p_df.Julian_Date <= (bounds[0])].index)
		p_df = p_df.drop(p_df[p_df.Julian_Date >= (bounds[1])].index)

	p_df = p_df.dropna()

	if plot_timeseries == True:
		plt.plot(p_df['Latitude'], p_df['Longitude'], 'ro', alpha=0.4)
		print p_df.head(2)

	return p_df

def clean_bw_tdgp(filepath, bounds=None, plot_timeseries=False):
	'''
	Clean textfiles logged by the prototype of Beckett and William.
	Inputs:
		filepath (string): path to the file from which to copy
		bounds (list of floats): Julian dates on which to truncate data
		plot_timeseries (bool): whether to plot a timeseries of the cleaned CTD data
	'''
	td_df = pd.read_table(filepath, delimiter=',', engine='c')
	print td_df.head(2)
	
	# #if bounds are present, only grab those bounds
	# if bounds is not None:
	# 	td_df = td_df.drop(td_df[td_df.Julian_Date <= (bounds[0])].index)
	# 	td_df = td_df.drop(td_df[td_df.Julian_Date >= (bounds[1])].index)

	# td_df = td_df.dropna()

	# if plot_timeseries == True:
	# 	plt.plot(td_df['Julian_Date'], td_df['CO2_ppm_adjusted'])
	# 	print td_df.head(2)
	
	# return td_df	

def geo_associate(df1, loc_df, keys):
	'''
	Geoassociates data from one frame to a frame with geo data.
	Input:
		df1 (pandas df): dataframe to be associated
		loc_df (pandas df): dataframe with geo location info
		keys (list of strings): hierarchical index names
	'''
	dat = df1.drop_duplicates(subset='Julian_Date', keep='last').set_index('Julian_Date')
	loc_dat = loc_df.drop_duplicates(subset='Julian_Date', keep='last').set_index('Julian_Date')
	temp = pd.concat([dat, loc_dat], axis=1, keys=keys)
	inter = temp.interpolate()
	ind = dat.index
	return inter.loc[ind]

def interp_all(dfs, keys, ind):
	jd_dfs = []
	for df in dfs:
		temp = df.drop_duplicates(subset='Julian_Date', keep='last').set_index('Julian_Date')
		jd_dfs.append(temp)

	all_temp = pd.concat(jd_dfs, axis=1, keys=keys)
	inter_temp = all_temp.interpolate()
	df_index = jd_dfs[ind].index
	return inter_temp.loc[df_index]


if __name__ == '__main__':
	###### 06.28/2018
	# save_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.28.2018/data/cleaned/'
	# query_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.28.2018/data/'
	# ctd_path = ['ctd/rbr_data/rbr_data_data.txt']
	# rpi_ctd_path = ['ctd/rbr_20180330013442.txt']
	# rpi_nit_path = ['nitrate/suna_20180330013640.txt']
	# nit_path = None
	# rpi_op_path = ['op/optode_20180330013534.txt']
	# airmar_path = ['airmar/airmar_20180330013612.txt']
	# gga_path = ['gga/2018-06-28/gga_2018-06-28_f0001.txt']
	# pix_path = None


	###### 06.29.2018
	# save_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.29.2018/data/cleaned/'
	# query_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.29.2018/data/'
	# ctd_path = ['ctd/rbr_data/rbr_data_data.txt']
	# rpi_ctd_path = ['ctd/rbr_20180330034916.txt']
	# rpi_nit_path = ['nitrate/suna_20180330034849.txt', 'nitrate/suna_20180330083032.txt', 'nitrate/suna_20180330083300.txt']
	# nit_path = ['nitrate/C0000132.CSV']
	# rpi_op_path = ['op/optode_20180330034739.txt', 'op/optode_20180330082905.txt']
	# airmar_path = ['airmar/airmar_20180330034652.txt', 'airmar/airmar_20180330082958.txt']
	# gga_path = ['gga/2018-06-29/gga_2018-06-29_f0002.txt']
	# pix_path = ['pix/2.BIN.gpx']

	###### 06.30.2018
	# save_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.30.2018/data/cleaned/'
	# query_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.30.2018/data/'
	# ctd_path = ['ctd/rbr_data/rbr_data_data.txt']
	# rpi_ctd_path = ['ctd/rbr_20180330093736.txt']
	# rpi_nit_path = ['nitrate/suna_20180330093830.txt']
	# # nit_path = ['nitrate/C0000132.CSV']
	# rpi_op_path = ['op/optode_20180330093807.txt']
	# airmar_path = ['airmar/airmar_20180330093911.txt']
	# gga_path = ['gga/2018-06-30/gga_2018-06-30_f0001.txt']

	######## 07.01.2018
	save_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/07.01.2018/data/cleaned/'
	query_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/07.01.2018/data/'
	ctd_path = ['ctd/rbr_data/rbr_data_data.txt']
	rpi_ctd_path = ['ctd/rbr_20180330132604.txt', 'ctd/rbr_20180330132706.txt']
	rpi_nit_path = ['nitrate/suna_20180330132643.txt']
	# nit_path = ['nitrate/C0000132.CSV']
	rpi_op_path = ['op/optode_20180330132615.txt']
	airmar_path = ['airmar/airmar_20180330132521.txt']
	gga_path = ['gga/2018-07-01/gga_2018-07-01_f0001.txt']
	pix_path = ['pix/22.BIN.gpx']

	geo_frame = 'pix'

	save_data = True
	geo_save_data = True
	save_all = True

	###############################
	##### Generate Dataframes #####
	###############################

	##### PIXHAWK #####
	m = []
	for p in pix_path:
		m.append(query_path+p)
	pix_df = clean_pix(m, bounds=None, plot_timeseries=True)
	if geo_frame == 'pix':
		start = np.sort(pix_df['Julian_Date'].values)[0] #2458299.18039
		end = np.sort(pix_df['Julian_Date'].values)[-1] #2458299.41531


	##### Airmar #####
	m = [] 
	for p in airmar_path:
		m.append(query_path+p)
	if geo_frame == 'pix':
		airmar_df = clean_airmar(m, bounds=None, plot_timeseries=True, offset=0.0)
	else:
		airmar_df = clean_airmar(m, bounds=None, plot_timeseries=True, offset=0.0)
		start = np.sort(airmar_df['Julian_Date'].values)[0] #2458299.18039
		end = np.sort(airmar_df['Julian_Date'].values)[-1] #2458299.41531

	print start, end

	##### CTD #####
	# m = []
	# for p in rpi_ctd_path:
	# 	m.append(query_path+p)
	# rpi_ctd_df = clean_rpi_ctd(m, bounds=[start,end], plot_timeseries=True) 

	m = []
	for p in ctd_path:
		m.append(query_path+p)
	ctd_df = clean_rbr_ctd(m, bounds=[start,end], plot_timeseries=False)

	##### NITRATE #####
	m = []
	for p in rpi_nit_path:
		m.append(query_path+p)
	rpi_nit_df = clean_rpi_nitrate(m, bounds=[start,end], plot_timeseries=False, day=1, month=7, year=2018)

	##### OPTODE #####
	m = []
	for p in rpi_op_path:
		m.append(query_path+p)
	# 06.29.2018 offset: 2440679.255
	# 06.30.2018 offset: 2440680.11575
	rpi_op_df = clean_rpi_optode(m, offset=2440680.893, bounds=[start,end], plot_timeseries=False)

	##### GGA #####
	m = []
	for p in gga_path:
		m.append(query_path+p)
	gga_df = clean_gga(m, offset_time=-0.0034, bounds=[start,end], plot_timeseries=False)

	# ##### TDGP #####
	# # please note! this is not yet completed; pending information from Beckett and William
	# bw_tdgp_data = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.28.2018/data/tdgp_beck/bw_tdgp.txt'
	# bw_tdgp_df = clean_bw_tdgp(bw_tdgp_data, bounds=[start,end], plot_timeseries=True)

	###############################
	##### Write Frames to File ####
	###############################
	if save_data == True:
		ctd_df.to_csv(save_path+'rbr.csv')
		# rpi_ctd_df.to_csv(save_path+'pi_rbr.csv')
		rpi_nit_df.to_csv(save_path+'nit.csv')
		rpi_op_df.to_csv(save_path+'op.csv')
		airmar_df.to_csv(save_path+'airmar.csv')
		gga_df.to_csv(save_path+'gga.csv')
		pix_df.to_csv(save_path+'pix.csv')

	###############################
	##### Geolocate the Sets ######
	# ###############################
	if geo_frame == 'pix':
		ctd_geo_df = geo_associate(ctd_df, pix_df, keys=['ctd', 'pix'])
		# rpi_ctd_geo_df = geo_associate(rpi_ctd_df, rpi_airmar_df, keys=['ctd', 'airmar'])
		nit_geo_df = geo_associate(rpi_nit_df, pix_df, keys=['nit', 'pix'])
		op_geo_df = geo_associate(rpi_op_df, pix_df, keys=['op', 'pix'])
		gga_geo_df = geo_associate(gga_df, pix_df, keys=['gga', 'pix'])
	else:
		ctd_geo_df = geo_associate(ctd_df, airmar_df, keys=['ctd', 'airmar'])
		# rpi_ctd_geo_df = geo_associate(rpi_ctd_df, rpi_airmar_df, keys=['ctd', 'airmar'])
		nit_geo_df = geo_associate(rpi_nit_df, airmar_df, keys=['nit', 'airmar'])
		op_geo_df = geo_associate(rpi_op_df, airmar_df, keys=['op', 'airmar'])
		gga_geo_df = geo_associate(gga_df, airmar_df, keys=['gga', 'airmar'])


	if geo_save_data == True:
		ctd_geo_df.to_csv(save_path+'geo_rbr.csv')
		# rpi_ctd_geo_df.to_csv(save_path+'geo_pi_rbr.csv')
		nit_geo_df.to_csv(save_path+'geo_nit.csv')
		op_geo_df.to_csv(save_path+'geo_op.csv')
		gga_geo_df.to_csv(save_path+'geo_gga.csv')

	###############################
	##### Create a Common Frame ###
	###############################
	if geo_frame == 'pix':
		indx = 4
	else:
		indx = 3
	all_df = interp_all([ctd_df, rpi_nit_df, rpi_op_df, airmar_df, pix_df, gga_df], keys=['ctd', 'nit', 'op', 'airmar', 'pix', 'gga'], ind=indx)

	if save_all == True:
		all_df.to_csv(save_path+'full_interp.csv')


	plt.show()



