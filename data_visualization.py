# !/usr/env/python

'''
A script for plotting data of interest for the Cambridge Bay collections.
'''

import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import pandas as pd


def viz_3d(x, y, z, c, label, title, vmin=None, vmax=None):
    if vmin==None:
        vmin = np.nanmin(c)
    if vmax==None:
        vmax = np.nanmax(c)
    
    m = plt.figure(figsize=(10,7))
    ax = m.add_subplot(111, projection='3d')
    cmap = plt.cm.viridis
    points = ax.scatter(x,y,z, c=c, s=5, alpha=0.5, lw=0, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = m.colorbar(points)
    cbar.set_label(label)
    m.suptitle(title, fontsize=18)
    ax.set_xlabel('Latitude', fontsize=18)
    ax.set_ylabel('Longitude', fontsize=18)
    ax.set_zlabel('Depth', fontsize=18)
    plt.savefig(label+'_3dviz.png')

def viz_cross(x1, x2, y, c, label, title, vmin=None, vmax=None):
    if vmin==None:
        vmin = np.nanmin(c)
    if vmax==None:
        vmax = np.nanmax(c)
    
    fig, ax = plt.subplots(1,2)
    
    cmap = plt.cm.viridis
    plt.suptitle(title, fontsize=18)
    
    points = ax[0].scatter(x1,y, c=c, s=10, alpha=0.5, lw=0, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].set_ylabel('Depth', fontsize=18)
    ax[0].set_xlabel('Longitude', fontsize=18)
    
    points = ax[1].scatter(x2,y, c=c, s=10, alpha=0.5, lw=0, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].set_xlabel('Latitude', fontsize=18)
    
    cbar = fig.colorbar(points)
    cbar.set_label(label)
    plt.savefig(label+'_cross_section.png')

def viz_top(x1, x2, c, label, title, vmin=None, vmax=None):
    if vmin==None:
        vmin = np.nanmin(c)
    if vmax==None:
        vmax = np.nanmax(c)
    
    fig = plt.figure(figsize=(10,7))
    
    cmap = plt.cm.viridis
    plt.suptitle(title, fontsize="18")
    
    points = plt.scatter(x1,x2, c=c, s=10, alpha=0.5, lw=0, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.ylabel('Latitude', fontsize=18)
    plt.xlabel('Longitude', fontsize=18)
    
    cbar = fig.colorbar(points)
    cbar.set_label(label)
    plt.savefig(label+'_flat_aerial.png')

def viz_layers(df, target, scale, label, title, vmin=None, vmax=None):
    if vmin==None:
        vmin = np.nanmin(df[target[0]][target[1]].values)
    if vmax==None:
        vmax = np.nanmax(df[target[0]][target[1]].values)
    
    cmap = plt.cm.viridis
    
    num_layers = int(np.nanmax(df['ctd']['Depth'].values)/scale)
    fig, ax = plt.subplots(num_layers,1, sharex=True, sharey=True, figsize=(6,num_layers*7))
    plt.suptitle(title, fontsize="18")

    for i in range(0,num_layers):
        lower = i*scale
        upper = (i+1)*scale
        temp_df = df[(df['ctd']['Depth'] < upper) & (df['ctd']['Depth'] >= lower)]
        points = ax[i].scatter(temp_df['pix']['Longitude'], temp_df['pix']['Latitude'], c=temp_df[target[0]][target[1]], s=10, alpha=0.5, lw=0, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i].set_title(str(lower) + " to " + str(upper) + " m", fontsize=18)

    plt.savefig(label+'_layered_aerial.png')


def create_3d_viz(df, insts, labels):
	for inst, l in zip(insts, labels):
		x = df['pix']['Latitude']
		y = df['pix']['Longitude']
		z = -df['ctd']['Depth']
		c = df[inst][l]
		label = l
		title = l
		viz_3d(x, y, z, c, label, title)

def create_cross_viz(df, insts, labels):
	for inst, l in zip(insts, labels):
		x1 = df['pix']['Longitude']
		x2 = df['pix']['Latitude']
		y = -df['ctd']['Depth']
		c = df[inst][l]
		label = l
		title = l
		viz_cross(x1, x2, y, c, label, title)

def create_aerial_overview(df, insts, labels, depth=True):
	for inst, l in zip(insts, labels):
		x1 = df['pix']['Longitude']
		x2 = df['pix']['Latitude']
		c = df[inst][l]
		label = l
		title = l

		viz_top(x1, x2, c, label, title)
		if depth == True:
			viz_layers(df, [inst, l], 0.5, label, title)



if __name__ == '__main__':
	# import the data of interest
	#6.29.2018
	base_path = '/media/vpreston/My Passport/Cambridge-Bay-06.2018/06.29.2018/data/cleaned/'
	all_data = 'full_interp.csv'
	ctd_geo = 'geo_rbr.csv'
	nit_geo = 'geo_nit.csv'
	op_geo = 'geo_op.csv'
	gga_geo = 'geo_gga.csv'


	# make the dataframes
	all_df = pd.read_table(base_path+all_data, delimiter=',', header=[0,1])
	ctd_df = pd.read_table(base_path+ctd_geo, delimiter=',', header=[0,1])
	gga_df = pd.read_table(base_path+gga_geo, delimiter=',', header=[0,1])
	nit_df = pd.read_table(base_path+nit_geo, delimiter=',', header=[0,1])
	op_df = pd.read_table(base_path+op_geo, delimiter=',', header=[0,1])

	###### 3D PLOTS ######
	insts = ['ctd', 'gga', 'gga', 'ctd']
	labels = ['Temperature', 'CH4_ppm_adjusted', 'CO2_ppm_adjusted', 'Salinity']

	# Lat, Lon, Depth, Prop
	create_3d_viz(all_df, insts, labels)

	# Cross Sections
	create_cross_viz(all_df, insts, labels)

	# Aerial Slices
	create_aerial_overview(all_df, insts, labels)

	###### 2D PLOTS ######
	insts = ['op']
	labels = ['O2Concentration']
	create_aerial_overview(op_df, insts, labels, depth=False)

	insts = ['nit']
	labels = ['data']
	create_aerial_overview(nit_df, insts, labels, depth=False)



	plt.show()

