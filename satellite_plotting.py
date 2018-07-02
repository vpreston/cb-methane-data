# !/usr/bin/env python

import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# map = Basemap(llcrnrlon=-105.034,llcrnrlat=69.105,urcrnrlon=-104.995,urcrnrlat=69.135, projection='cyl')
#http://server.arcgisonline.com/arcgis/rest/services
# map = Basemap(llcrnrlon=-10.5,llcrnrlat=35,urcrnrlon=4.,urcrnrlat=44.,
#              resolution='i', projection='cyl', lat_0 = 39.5, lon_0 = -3.25)

map = Basemap(llcrnrlon=-105.05,llcrnrlat=69.1,urcrnrlon=-105.,urcrnrlat=69.15,
             resolution='i', projection='cyl')

map.arcgisimage(service='World_Topo_Map', xpixels=400, verbose= True)

# map = Basemap(llcrnrlon=-10.5,llcrnrlat=35,urcrnrlon=4.,urcrnrlat=44.,
#              resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

# map = Basemap(llcrnrlon=-106.,llcrnrlat=69.0,urcrnrlon=-104.,urcrnrlat=69.5,
#              resolution='i', projection='tmerc', lat_0 = 69.25, lon_0 = -105.)

# map.drawmapboundary(fill_color='aqua')
# map.fillcontinents(color='coral',lake_color='aqua')
# map.drawcoastlines()
# map.bluemarble(scale=1.0)

plt.show()