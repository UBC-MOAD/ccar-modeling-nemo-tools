'''
============================================
NEMO_tools
--------------------------------------------
A collection of useful functions for NEMO
                ----- CCAR Modelling Team
============================================
2014/12/11 File created
2014/12/12 Add "plot_Arctic_LandCover"
2014/12/14 Add "reporj_xygrid", "reporj_NEMOgrid"
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import cm
from mpl_toolkits.basemap import Basemap

def reporj_NEMOgrid(raw_x, raw_y, raw_data, nav_lon, nav_lat, method='nearest'):
    '''
    =======================================================================
    Reproject irregular data to a xy grid, works for various cases
    (See Nancy's IPy notebook in SalishSea/analysis )
                            ----- created on 2014/12/14, Yingkai (Kyle) Sha    
    -----------------------------------------------------------------------
    data_interp = reporj_xygrid(...)
    -----------------------------------------------------------------------
    Input:
            raw_data: input data, N*M 2-D array.
            raw_x: longitude grid, N*M 2-D array.
            raw_y: latitude grid, N*M 2-D array.
            nav_lon: NEMO latitude grid.
            nav_lat: NEMO longitude grid.
            methods: interpolation methods ('nearest', 'linear',  'cubic')
    Output:
            data_interp: interpolated data
    =======================================================================
    '''
    from scipy.interpolate import griddata
    LatLonPair=(raw_x.flatten(), raw_y.flatten())
    data_interp = griddata(LatLonPair, raw_data.flatten(), (nav_lon, nav_lat), method=method)
    return data_interp

def reporj_xygrid(raw_x, raw_y, raw_data, xlim, ylim, res):
    
    '''
    =======================================================================
    Reproject irregular data to a xy grid, works for various cases
                            ----- created on 2014/12/14, Yingkai (Kyle) Sha    
    -----------------------------------------------------------------------
    d_array, x_array, y_array, bin_count = reporj_xygrid(...)
    -----------------------------------------------------------------------
    Input:
            raw_data: irregular data, N*M 2-D array.
            raw_x: longitude info. N*M 2-D array.
            raw_y: latitude info. N*M 2-D array.
            xlim: range of longitude, a list.
            ylim: range of latitude, a list.
            res: resolution, 2 elements for x- and y-axis.
    Output:
            d_array: reprojected data.
            x_array: reprojected longitude.
            y_array: reprojected latitude.
            bin_count: how many raw data point included in a reprojected grid.
    Note:
            function do not performs well if "res" is higher than original.
            size of "raw_data", "raw_x", "raw_y" must agree.
    =======================================================================
    '''
    import numpy as np
    
    x_bins=np.arange(xlim[0], xlim[1], res[0])
    y_bins=np.arange(ylim[0], ylim[1], res[1])
    x_indices=np.searchsorted(x_bins, raw_x.flat, 'right')
    y_indices=np.searchsorted(y_bins, raw_y.flat, 'right')       
    y_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    x_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    d_array=np.zeros([len(y_bins), len(x_bins)], dtype=np.float)
    bin_count=np.zeros([len(y_bins), len(x_bins)], dtype=np.int)
    
    for n in range(len(y_indices)): #indices
        bin_row=y_indices[n]-1 # '-1' is because we call 'right' in np.searchsorted.
        bin_col=x_indices[n]-1
        bin_count[bin_row, bin_col] += 1
        x_array[bin_row, bin_col] += raw_x.flat[n]
        y_array[bin_row, bin_col] += raw_y.flat[n]
        d_array[bin_row, bin_col] += raw_data.flat[n]
                   
    for i in range(x_array.shape[0]):
        for j in range(x_array.shape[1]):
            if bin_count[i, j] > 0:
                x_array[i, j]=x_array[i, j]/bin_count[i, j]
                y_array[i, j]=y_array[i, j]/bin_count[i, j]
                d_array[i, j]=d_array[i, j]/bin_count[i, j] 
            else:
                d_array[i, j]=np.nan
                x_array[i, j]=np.nan
                y_array[i, j]=np.nan
                
    return d_array, x_array, y_array, bin_count

def plot_NEMO_grid(lon, lat, color='k', linewidth=0.5, relax=1, location='north'):
    '''
    =======================================================================
        Plot NEMO grid, works for: 
                                ORCA2 tripolar grid
                                Xianming's grid
                                
                            ----- created on 2014/12/11, Yingkai (Kyle) Sha        
    -----------------------------------------------------------------------
        fig, axes=draw_ORCA2_grid(...)
    -----------------------------------------------------------------------
        Input:
                lon, lat : longitude and latitude records, usually nav_lon, nav_lat
                color    : color of line object
                linewidth: width of line object
                relax    : draw every 2 grid line (when =2)
                location : for north hemisphere (= north)
                          for south hemisphere (= south)
                          for both (= both)
        Output:
                fig, axes: figure and axes objects
    =======================================================================            
    '''
    if location=='both':
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        #fig, axes = plt.figure(figsize=(10, 10))
    # north hemisphere <----- left
    if (location=='both') or (location=='north'):
        if location=='both':
            ax=axes[0]
        else:
            ax=axes # <----- does not support indexing
        proj=Basemap(projection='npstere', resolution='l', boundinglat=0, lon_0=90, round=True, ax=ax)
        # coastline, maskland
        proj.drawcoastlines(linewidth=1.5, linestyle='-', color='k', zorder=3)
        proj.drawlsmask(land_color=[0.5, 0.5, 0.5], ocean_color='None', lsmask=None, zorder=2)
        # lon, lat -----> x, y coordinates in basemap
        x, y=proj(lon[::relax, ::relax], lat[::relax, ::relax])
        # plot
        proj.plot(x.T, y.T, color=color, linewidth=linewidth)
        proj.plot(x, y, color=color, linewidth=linewidth)
    # south hemisphere <----- right
    if (location=='both') or (location=='south'): 
        if location=='both':
            ax=axes[1]
        else:
            ax=axes # <----- does not support indexing
        proj=Basemap(projection='spstere', resolution='l', boundinglat=0, lon_0=90, round=True, ax=ax)
        ax=plt.gca()
        proj.drawcoastlines(linewidth=1.5, linestyle='-', color='k', zorder=3)
        proj.drawlsmask(land_color=[0.5, 0.5, 0.5], ocean_color='None', lsmask=None, zorder=2)
        x, y=proj(lon, lat)
        proj.plot(x[:, :].T, y[:, :].T, color=color, linewidth=linewidth)
        proj.plot(x[:, :], y[:, :], color=color, linewidth=linewidth)
    return fig, axes

def plot_NEMO_Arctic(lon, lat, lat0, var, clev, CMap, var_name='variable'):
    '''
    =======================================================================
        PLot data (contours) on Arctic, works for various cases
                            ----- created on 2014/12/11, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        fig, ax, proj = plot_NEMO_Arctic(...)
    -----------------------------------------------------------------------
        Input:
                lon, lat: longitude and latitude records, usually nav_lon, nav_lat
                lat0    : bounding latitude for Arctic
                var     : variable
                clev    : number of contours or specific values of contours
                var_name: name and unit show on the colorbar
                CMap    : colormap <----- (e.g. plt.cm.jet)
        Output:
                fig, ax, proj: figure, axis, basemap object
    =======================================================================    
    '''
    fig=plt.figure(figsize=(14, 14))
    proj=Basemap(projection='npstere', resolution='l', boundinglat=lat0, lon_0=90, round=True)
    ax=plt.gca()
    # parallels & meridians
    parallels=np.arange(-90, 90, 15)
    meridians=np.arange(0, 360, 60)
    proj.drawparallels(parallels, labels=[1, 1, 1, 1],\
                      fontsize=10, latmax=90)
    proj.drawmeridians(meridians, labels=[1, 1, 1, 1],\
                      fontsize=10, latmax=90)
    # coastline, maskland
    proj.drawcoastlines(linewidth=1.5, linestyle='-', color='k', zorder=3)
    proj.drawlsmask(land_color=[0.5, 0.5, 0.5], ocean_color='None', lsmask=None, zorder=2)
    # lon, lat -----> x, y coordinates in basemap
    x, y=proj(lon, lat)
    # plot
    CS=proj.contourf(x, y, var, clev, cmap=CMap, extend='both')
    proj.contour(x, y, var, clev, colors='k', linewidths=2.0)
    CBar=proj.colorbar(CS, location='right', size='5%', pad='10%')
    CBar.set_label(var_name, fontsize=14, fontweight='bold')
    CBar.ax.tick_params(axis='y', length=0)
    return fig, ax, proj

def plot_Arctic_LandCover(lon, lat, lat0, var, clev, CMap, var_name='variable'):
    '''
    =======================================================================
        PLot classification of Arctic hydrological basins.
        Designed for STN-30p data, data source:
            University of New Hampshire, Water Systems Research Group
                Composite Runoff Fields v1.0
                [http://www.grdc.sr.unh.edu/html/Data/index.html]
                
                            ----- created on 2014/12/12, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        fig, ax, proj = plot_Arctic_LandCover(...)
    -----------------------------------------------------------------------
        Input:
                lon, lat: longitude and latitude records, usually nav_lon, nav_lat
                lat0    : bounding latitude for Arctic
                var     : variable
                clev    : numbers of hydrological basins + 1, because I use contours
                var_name: name and unit show on the colorbar
                CMap    : colormap <----- (e.g. plt.cm.jet)
        Output:
                fig, ax, proj: figure, axis, basemap object
    =======================================================================    
    '''
    fig=plt.figure(figsize=(14, 14))
    proj=Basemap(projection='npstere', resolution='l', boundinglat=lat0, lon_0=90, round=True)
    ax=plt.gca()
    # parallels & meridians
    parallels=np.arange(-90, 90, 15)
    meridians=np.arange(0, 360, 60)
    proj.drawparallels(parallels, labels=[1, 1, 1, 1],\
                      fontsize=10, latmax=90, linewidth=0)
    proj.drawmeridians(meridians, labels=[1, 1, 1, 1],\
                      fontsize=10, latmax=90, linewidth=0)
    # coastline, maskland
    proj.drawcoastlines(linewidth=1.5, linestyle='-', color='k', zorder=3)
    # lon, lat -----> x, y coordinates in basemap
    x, y=proj(lon, lat)
    # plot
    CS=proj.contourf(x, y, var, clev, cmap=CMap)
    proj.pcolor(x, y, var, vmin=clev[0], vmax=clev[-1], cmap=CMap)
    proxy = [plt.Rectangle((0, 0), 1, 1, fc = pc.get_facecolor()[0]) for pc in CS.collections]
    plt.legend(proxy, ["Arctic Archipelago", \
                       "Arctic subocean",    \
                       "Baffin Bay",         \
                       "Barents Sea",        \
                       "Beaufort Sea",       \
                       "Bering Strait",      \
                       "Chukchi Sea",        \
                       "East Siberian Sea",  \
                       "Foxe Basin",         \
                       "Hudson Bay",         \
                       "Hudson Strait",      \
                       "Kara Sea",           \
                       "Laptev Sea",         \
                       "Norwegian Sea",      \
                       "Ob Internal Basins", \
                       "South Greenland Sea",\
                       "Greenland Sea"],     \
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1)
    #CBar=proj.colorbar(CS, location='right', size='5%', pad='10%')
    #CBar.set_label(var_name, fontsize=14, fontweight='bold')
    #CBar.ax.tick_params(axis='y', length=0)
    return fig, ax, proj


def plot_Arctic_LandCover_RAW(lon, lat, lat0, var, clev, CMap):
    '''
    =======================================================================
        Simplified version of "plot_Arctic_LandCover" 
        (See "plot_Arctic_LandCover")
                
                            ----- created on 2014/12/12, Yingkai (Kyle) Sha
    =======================================================================    
    '''
    fig=plt.figure(figsize=(14, 14))
    proj=Basemap(projection='npstere', resolution='l', boundinglat=lat0, lon_0=90, round=True)
    ax=plt.gca()
    # parallels & meridians
    #parallels=np.arange(-90, 90, 10)
    #meridians=np.arange(0, 360, 5)
    #proj.drawparallels(parallels, labels=[0, 0, 0, 0],\
    #                  fontsize=10, latmax=90, linewidth=0.1, linestyle='-')
    #proj.drawmeridians(meridians, labels=[0, 0, 0, 0],\
    #                  fontsize=10, latmax=90, linewidth=0.1, linestyle='-')
    #proj.drawcoastlines(linestyle='-', linewidth=0.1, color='k', zorder=3)
    x, y=proj(lon, lat)
    CS=proj.pcolor(x, y, var, vmin=clev[0], vmax=clev[-1], cmap=CMap)
    #CBar=proj.colorbar(CS, location='right', size='5%', pad='10%')
    return fig, ax, proj


    
