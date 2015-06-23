'''
============================================
NEMO_tools
--------------------------------------------
A collection of useful functions for NEMO
                ----- CCAR Modelling Team
                -----   Yingkai (Kyle) Sha
============================================
2014/12/11 File created
2014/12/12 Add "plot_Arctic_LandCover"
2014/12/14 Add "reporj_xygrid", "reporj_NEMOgrid"
2014/12/18 Add "pcolor_Arctic", "delete_edge"
2014/12/22 Add "map_Arctic", rewrite all plotting functions
2014/12/30 Add "land_mask"
2015/06/22 Several new functions added 
'''
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import cm
from mpl_toolkits.basemap import Basemap

## =============== Timeseries analysis =============== ##

def bin_monmean(dt, data):
    '''
    =======================================================================
    Convert data in days to monthly mean series
                            ----- created on 2014/12/25, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        data_dt, series, bin_count = bin_monmean(...)
    -----------------------------------------------------------------------
    Input:
            dt: A list of original data's datetime.datetime objects
            data: original data
    Output:
            data_dt: A list of monmean datetime.datetime objects
            data: monmean data
            bin_count: how many points fall into each months, 
                        data[i]=np.nan when bin_count[i]=0.
    ======================================================================= 
    '''
    import datetime
    import numpy as np
    from dateutil.relativedelta import relativedelta
    
    dt=sorted(dt) # increase order
    # Calculate the length of series
    L=(dt[-1].year-dt[0].year+1)*12
    series=np.zeros(L)
    bin_count=np.zeros(L)
    # Bin data points
    for i in range(len(dt)):
        hit=(dt[i].year-dt[0].year)*12+dt[i].month
        series[hit-1] += data[i] # "-1" because it is Python
        bin_count[hit-1] += 1 # count    
    for i in range(L):
        if bin_count[i] > 0:
            series[i]=series[i]/bin_count[i]
        if bin_count[i] == 0:
            series[i]=np.nan            
    # generate a corresponding datetime series
    data_dt=[datetime.datetime(dt[0].year, 01, 01)]
    step=relativedelta(months=1)    
    for i in range(L-1): # don't know why but it needs L-1, or the size will missmatch
        temp=data_dt[i]
        temp += step
        data_dt.append(temp)
    
    return data_dt, series, bin_count
    
def bin_season_cycle(dt, data):
    '''
    =======================================================================
    bin daily data into 12 months
                            ----- created on 2014/12/25, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        series, bin_count = bin_season_cycle(...)
    -----------------------------------------------------------------------
    Input:
            dt: A list of original data's datetime.datetime objects
            data: original data
    Output:
            data: data in 12 months
            bin_count: how many points fall into each months, 
                        data[i]=np.nan when bin_count[i]=0.
    ======================================================================= 
    '''
    #import datetime
    import numpy as np
    #from dateutil.relativedelta import relativedelta
    
    dt=sorted(dt) # increase order
    series=np.zeros(12)
    bin_count=np.zeros(12)
    # Bin data points
    for i in range(len(dt)):
        hit=dt[i].month
        series[hit-1] += data[i] # "-1" because it is Python
        bin_count[hit-1] += 1 # count    
    for i in range(12):
        if bin_count[i] > 0:
            series[i]=series[i]/bin_count[i]
        if bin_count[i] == 0:
            series[i]=np.nan        
    return series, bin_count
    
def int_between(begin, end, num_between):
    '''
    =======================================================================
    Use linear interpolation get values (equally distributed) between knowns
                            ----- created on 2015/05/08, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        out = int_between(...)
    -----------------------------------------------------------------------
    Input:
            begin: x0
            end: x1
            num_between: how many points you want
    ======================================================================= 
    '''
    from scipy.interpolate import interp1d
    f = interp1d([0, 1], [begin, end])
    return f(np.linspace(0, 1, num_between+2))[1:-1]    
    
## =============== Grid analysis =============== ##

def nearest_search(nav_lon, nav_lat, lons, lats):
    '''
    =======================================================================
    Get the nearest grid point of your lons, lats
                            ----- created by Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        ind_x, ind_y = nearest_search(...)
    -----------------------------------------------------------------------
    Input:
            begin: x0
            end: x1
            num_between: how many points you want
    ======================================================================= 
    '''
    from scipy.spatial import cKDTree
    combined_x_y_arrays = np.dstack([nav_lon.ravel(), nav_lat.ravel()])[0]
    points_list = list(np.array([lons.T, lats.T]).T)
    #
    mytree = cKDTree(combined_x_y_arrays)
    dist, index_flat = mytree.query(points_list)
    x, y = np.unravel_index(index_flat, nav_lon.shape)
    return x, y

def delete_edge(data, order=[1, 1, 1, 1]):
    
    ''' 
    =======================================================================
    Delete the edge of input data, initially designed for NEMO GYRE
                            ----- created on 2014/12/18, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
    data_output = delete_edge(...)
    -----------------------------------------------------------------------
    Input:
            data: input data, originally designed for Lat*Lon;
            order: how many rows/columns you'd like to delete in 4 edges;

                order(2)
              -----------
             |           |
    order(1) |           | order(3)
             |           |
              -----------
                order(4)

    Note: 
            delete means NaN. You can mask NaN for plot.
    =======================================================================
    '''    
    data[:, 0:order[0]]=np.nan
    data[0:order[1], :]=np.nan
    data[:, (np.size(data, 1)-order[2]):np.size(data, 1)]=np.nan
    data[(np.size(data, 0)-order[3]):np.size(data, 0), :]=np.nan

    return data

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
    Note:
            Omini-functioning, works for both regular and irregular grid
            but a bit slow.
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
       
def mask_land(lon, lat, data, hit='ORCA2_Arctic'):
    '''
    =======================================================================
    Mask land area based on NEMO's landmask file.
                            ----- created on 2014/12/29, Yingkai (Kyle) Sha    
    -----------------------------------------------------------------------
    data = mask_land(...)
    -----------------------------------------------------------------------
    Input:
            lon, lat: longitude/latitude records for original data
            data: data to be masked
            hit: different kind of mask
    Output:
            data: masked array
    =======================================================================
    '''
    from scipy.io import loadmat
    if hit == 'ORCA2_Arctic':
        mask_obj=loadmat('ORCA2_Landmask_Arctic.mat')
        mask_data=mask_obj['ORCA2_Landmask_Arctic']
        ref_lat=mask_obj['nav_lat']
        ref_lon=mask_obj['nav_lon']
        
    mask_interp=reporj_NEMOgrid(ref_lon, ref_lat, mask_data, lon, lat, method='Nearest')
    data=np.ma.masked_where(mask_interp==1, data)

    return data
    

def plot_NEMO_grid(lon, lat, bound_lat=0, color='k', linewidth=0.5, relax=1, location='north'):
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
                bound_lat: Latitude boundary, e.g. =45 means starts from 45N/S for North/South
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
        proj=Basemap(projection='npstere', resolution='l', boundinglat=bound_lat, lon_0=90, round=True, ax=ax)
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
        proj=Basemap(projection='spstere', resolution='l', boundinglat=-bound_lat, lon_0=90, round=True, ax=ax)
        ax=plt.gca()
        proj.drawcoastlines(linewidth=1.5, linestyle='-', color='k', zorder=3)
        proj.drawlsmask(land_color=[0.5, 0.5, 0.5], ocean_color='None', lsmask=None, zorder=2)
        x, y=proj(lon, lat)
        proj.plot(x[:, :].T, y[:, :].T, color=color, linewidth=linewidth)
        proj.plot(x[:, :], y[:, :], color=color, linewidth=linewidth)
    return fig, axes

    
def map_Arctic(lon, lat, lat0, hit):
    '''
    =======================================================================
        Create a map of Arctic in Stereographic Proj., returns the object 
        of figure, axis and basemap frame.
                            ----- created on 2014/12/22, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        fig, ax, proj = map_Arctic(...)
    -----------------------------------------------------------------------
        Input:
                lon, lat: longitude and latitude records, usually nav_lon, nav_lat
                lat0    : bounding latitude for Arctic
                hit     : (=1): maskland + draw parallels/meridians 
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
    lw=0
    if hit==1:
        lw=0.5
    proj.drawparallels(parallels, labels=[1, 1, 1, 1],\
                      fontsize=10, latmax=90, linewidth=lw)
    proj.drawmeridians(meridians, labels=[1, 1, 1, 1],\
                      fontsize=10, latmax=90, linewidth=lw)
    # coastline, maskland
    proj.drawcoastlines(linewidth=1.5, linestyle='-', color='k', zorder=3)
    if hit==1:
        proj.drawlsmask(land_color=[0.5, 0.5, 0.5], ocean_color='None', lsmask=None, zorder=2)
    return fig, ax, proj
    
def contourf_Arctic(lon, lat, lat0, var, clev, CMap, var_name='variable', hit=1):
    '''
    =======================================================================
        PLot data (contours) on Arctic, works for various cases
                            ----- created on 2014/12/11, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        fig, ax, proj = contourf_Arctic(...)
    -----------------------------------------------------------------------
        Input:
                lon, lat: longitude and latitude records, usually nav_lon, nav_lat
                lat0    : bounding latitude for Arctic
                var     : variable
                clev    : number of contours or specific values of contours
                var_name: name and unit show on the colorbar
                CMap    : colormap <----- (e.g. plt.cm.jet)
                hit     : (=1): maskland + draw parallels/meridians 
        Output:
                fig, ax, proj: figure, axis, basemap object
    =======================================================================    
    '''
    fig, ax, proj = map_Arctic(lon, lat, lat0, hit)
    x, y=proj(lon, lat)
    # plot
    CS=proj.contourf(x, y, var, clev, cmap=CMap, extend='both')
    proj.contour(x, y, var, clev, colors = ([0.5, 0.5, 0.5],), linewidths=1.0)
    CBar=proj.colorbar(CS, location='right', size='5%', pad='10%')
    CBar.set_label(var_name, fontsize=14, fontweight='bold')
    CBar.ax.tick_params(axis='y', length=0)
    return fig, ax, proj
    
def pcolor_Arctic(lon, lat, lat0, var, var_lim, CMap, var_name='variable', hit=1):
    '''
    =======================================================================
        PLot data (contours) on Arctic, works for various cases
                            ----- created on 2014/12/18, Yingkai (Kyle) Sha
    -----------------------------------------------------------------------
        fig, ax, proj = pcolor_Arctic(...)
    -----------------------------------------------------------------------
        Input:
                lon, lat: longitude and latitude records, usually nav_lon, nav_lat
                lat0    : bounding latitude for Arctic
                var     : variable
                var_lim : [vmin, vmax]
                var_name: name and unit show on the colorbar
                CMap    : colormap <----- (e.g. plt.cm.jet)
                hit     : (=1): maskland + draw parallels/meridians 
        Output:
                fig, ax, proj: figure, axis, basemap object
    =======================================================================    
    '''
    fig, ax, proj = map_Arctic(lon, lat, lat0, hit)
    x, y=proj(lon, lat)
    # plot
    CS=proj.pcolor(x, y, var, vmin=var_lim[0], vmax=var_lim[1], cmap=CMap)
    CBar=proj.colorbar(CS, location='right', size='5%', pad='10%')
    CBar.set_label(var_name, fontsize=14, fontweight='bold')
    CBar.ax.tick_params(axis='y', length=0)
    return fig, ax, proj

def plot_Arctic_LandCover(lon, lat, lat0, var, clev, regions, CMap, var_name='variable', hit=0):
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
                hit     : (=1): draw parallels/meridians
        Output:
                fig, ax, proj: figure, axis, basemap object
    =======================================================================    
    '''
    fig, ax, proj = map_Arctic(lon, lat, lat0, hit)
    x, y=proj(lon, lat)
    # plot
    CS=proj.contourf(x, y, var, clev, cmap=CMap)
    proj.pcolor(x, y, var, vmin=clev[0], vmax=clev[-1], cmap=CMap)
    proxy = [plt.Rectangle((0, 0), 1, 1, fc = pc.get_facecolor()[0]) for pc in CS.collections]
    LG=plt.legend(proxy, regions, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1); LG.draw_frame(False)  
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
    x, y=proj(lon, lat)
    CS=proj.pcolor(x, y, var, vmin=clev[0], vmax=clev[-1], cmap=CMap)
    #CBar=proj.colorbar(CS, location='right', size='5%', pad='10%')
    return fig, ax, proj


    
