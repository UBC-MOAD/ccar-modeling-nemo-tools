'''
============================================
NEMO_tools
--------------------------------------------
A collection of useful functions for NEMO
                ----- CCAR Modelling Team
============================================
2014/12/11 File created
'''
import numpy as np
import matplotlib.pyplot as plt
#from __future__ import division
#from __future__ import print_function
from mpl_toolkits.basemap import cm
from mpl_toolkits.basemap import Basemap

def plot_NEMO_grid(lon, lat, color='k', linewidth=0.5, relax=1, location='north'):
    '''
    =======================================================================
        Plot NEMO grid, works for: 
                                ORCA2 tripolar grid
                                Xianming's 7km grid
                                
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
    
    
