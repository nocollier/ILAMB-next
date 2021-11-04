import os
import glob
from Variable import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from Post import generate_plot_database

def is_spatially_aligned(a,b):
    """Are the lats and lons of a and b aligned?

    """
    if a.lat_name is None or b.lat_name is None: return False
    if a.lon_name is None or b.lon_name is None: return False
    if a.ds[a.lat_name].size != b.ds[b.lat_name].size: return False
    if a.ds[a.lon_name].size != b.ds[b.lon_name].size: return False
    if not np.allclose(a.ds[a.lat_name],b.ds[b.lat_name]): return False
    if not np.allclose(a.ds[a.lon_name],b.ds[b.lon_name]): return False
    return True

def pick_grid_aligned(r0,c0,r,c):
    """Pick variables for r and c such that they are grid aligned without recomputing if not needed.
    
    """
    if is_spatially_aligned(r0,c0): return r0,c0
    if (r is not None) and (c is not None):
        if is_spatially_aligned(r,c): return r,c
    return r0.nestSpatialGrids(c0)

def adjust_lon(a,b):
    """When comparing b to a, we need their longitudes uniformly in [-180,180) or [0,360).

    """
    if a.lon_name is None or b.lon_name is None: return a,b
    a360 = (a.ds[a.lon_name].min() >= 0)*(a.ds[a.lon_name].max() <= 360)
    b360 = (b.ds[b.lon_name].min() >= 0)*(b.ds[b.lon_name].max() <= 360)
    if a360 and not b360:
        b.ds[b.lon_name] = b.ds[b.lon_name] % 360
        b.ds = b.ds.sortby(b.lon_name)
    elif not a360 and b360:
        b.ds[b.lon_name] = (b.ds[b.lon_name]+180) % 360-180
        b.ds = b.ds.sortby(b.lon_name)
    return a,b

def sanitize_into_dataset(d):
    """Cleanup dictionaries for use in creation of a xarray dataset.

    """
    ds = {}

    # group dataarrays by unique sets of lats/lons and rename as needed
    lats = {}; lons = {}
    for key in d:
        v = d[key]
        if v.lat_name is None: continue
        if v.lon_name is None: continue
        lat = v.ds[v.lat_name]
        lon = v.ds[v.lon_name]
        if lat.size not in lats: lats[lat.size] = []
        if lon.size not in lons: lons[lon.size] = []
        lats[lat.size].append(key)
        lons[lon.size].append(key)
    assert len(lats) == len(lons)
    lat_name = "lat"; lon_name = "lon"
    for nlat,nlon in zip(sorted(lats.keys()),sorted(lons.keys())):
        assert(len(set(lats[nlat]).difference(set(lons[nlon])))==0)
        for name in lats[nlat]:
            v = d[name]
            da = v.ds[v.varname]
            ds[name] = da.rename({v.lat_name:lat_name,v.lon_name:lon_name})
        lat_name += "_"
        lon_name += "_"

    # group remaining dataaarays by unqiue sets of times and rename as needed
    times = {}
    for key in set(d.keys()).difference(set(ds.keys())):
        v = d[key]
        if 'time' not in v.ds[v.varname].dims: continue
        t = v.ds['time']
        if t.size not in times: times[t.size] = []
        times[t.size].append(key)
    t_name = "time"
    for nt in sorted(times.keys()):
        for name in times[nt]:
            v = d[name]
            da = v.ds[v.varname]
            ds[name] = da.rename({'time':t_name})
        t_name += "_"
        
    # make sure we dealt with everything
    assert(len(set(d.keys()).difference(set(ds.keys())))==0)
    keep = ['_FillValue','units','ilamb','analysis','region']
    for name in ds:
        ds[name].attrs = { key:val for key,val in ds[name].attrs.items() if key in keep }
        #ds[name].attrs['actual_range'] = [ds[name].min(),ds[name].max()]
        #ds[name].attrs['percentiles'] = list(ds[name].quantile([0.01,0.99]).to_numpy())
    ds = xr.Dataset(ds)
    return ds

def overall_score(df):
    """Adds a overall score as the mean of all scores in the input dataframe by region.

    """
    os = df[df.ScalarType=='score']
    model = os.Model.unique()[0]
    nr = len(os.Region.unique())
    os = os.groupby('Region').mean()
    df = pd.concat([df,pd.DataFrame({'Model'      : [model]*nr,
                                     'Region'     : list(os.index),
                                     'ScalarName' : ['Overall Score']*nr,
                                     'ScalarType' : ['score']*nr,
                                     'Units'      : ['1']*nr,
                                     'Data'       : list(os.Data)},
                                    columns=['Model','Region','ScalarName','ScalarType','Units','Data'])],
                   ignore_index=True)
    return df

def add_analysis_name(name,*args):
    """For each dictionary in args and each variable therein, add the analysis name to the attributes.

    """
    for a in args:
        for key in a:
            v = a[key]
            v.ds[v.varname].attrs['analysis'] = name
    return args

def ScoreBias(r0,c0,r=None,c=None,regions=[None]):
    """

    """
    v = r0.varname
    aname = "Bias"
    sdim = "site" if r0.sites() else "space"
    
    # period means on original grids
    rm0 = r0.integrate(dim='time',mean=True) if r0.temporal() else r0
    cm0 = c0.integrate(dim='time',mean=True) if c0.temporal() else c0

    # if we have temporal data, the normalizer is the std
    norm0 = r0.std(dim='time') if r0.ds['time'].size > 1 else rm0

    # interpolate to a nested grid
    rm,cm,norm = rm0.nestSpatialGrids(cm0,norm0)
    bias = cm-rm
    
    # do we have reference uncertainties?
    ru = rm.uncertainty()
    un = 0 if ru is None else ru.ds[ru.varname]
    
    # compute the bias score
    eps  = cm-rm    
    eps.ds[eps.varname] = (np.abs(eps.ds[eps.varname])-un).clip(0)
    eps.ds[eps.varname].attrs['units'] = cm.units()
    eps /= norm
    eps.ds[eps.varname] = np.exp(-eps.ds[eps.varname])

    # populate scalars over regions
    df = []
    for region in regions:
        s = rm0.integrate(dim=sdim,mean=True,region=region)
        df.append(['Reference',str(region),aname,'Period Mean','scalar',s.units(),float(s.ds[s.varname].values)])
        s = cm0.integrate(dim=sdim,mean=True,region=region)
        df.append(['model',str(region),aname,'Period Mean','scalar',s.units(),float(s.ds[s.varname].values)])
        s = bias.integrate(dim=sdim,mean=True,region=region)
        df.append(['model',str(region),aname,'Bias','scalar',s.units(),float(s.ds[s.varname].values)])
        s = eps.integrate(dim=sdim,mean=True,region=region)
        df.append(['model',str(region),aname,'Bias Score','score',s.units(),float(s.ds[s.varname].values)])
    df = pd.DataFrame(df,columns=['Model','Region','Analysis','ScalarName','ScalarType','Units','Data'])

    # collect output for intermediate files
    r_plot = {
        "timeint_of_%s"       % v: rm0,
    }
    if ru is not None: r_plot["uncertain"] = ru
    c_plot = {
        "timeint_of_%s"       % v: cm0,
        "bias_map_of_%s"      % v: bias,
        "biasscore_map_of_%s" % v: eps
    }
    r_plot,c_plot = add_analysis_name(aname,r_plot,c_plot)
    return r_plot,c_plot,df

def ScoreRMSE(r0,c0,r=None,c=None,regions=[None]):
    """

    """
    v = r0.varname
    aname = "RMSE"
    sdim = "site" if r0.sites() else "space"
    
    # validity checks
    if r0.ds['time'].size < 12: return {},{},[]

    # get normalizer and regrid
    norm0 = r0.std(dim='time')
    r,c,norm = r0.nestSpatialGrids(c0,norm0)

    # do we have reference uncertainties?
    ru = r.uncertainty()
    un = 0 if ru is None else ru.ds[ru.varname]

    # compute the RMSE error and score
    rmse = r.rmse(c)
    r = r.detrend(degree=0)
    c = c.detrend(degree=0)
    eps = c-r
    del c,r
    eps.ds[eps.varname] = (np.abs(eps.ds[eps.varname])-un).clip(0)**2
    eps.ds[eps.varname].attrs['units'] = "1"
    eps = eps.integrate(dim='time',mean=True)
    eps /= norm
    eps.ds[eps.varname] = np.exp(-eps.ds[eps.varname])    
    eps.ds[eps.varname].attrs['units'] = "1"
    
    # collect output for intermediate files
    r_plot = {}
    c_plot = {
        'rmse_map_of_%s'      % v: rmse,
        'rmsescore_map_of_%s' % v: eps
    }
    df = []
    for region in regions:
        name = 'spaceint_of_%s_over_%s' % (v,region)
        r_plot[name] = r0.integrate(sdim,mean=True,region=region)
        c_plot[name] = c0.integrate(sdim,mean=True,region=region)
        s = rmse.integrate(sdim,mean=True,region=region)
        df.append(['model',str(region),aname,'RMSE','scalar',s.units(),float(s.ds[s.varname].values)])
        s = eps.integrate(sdim,mean=True,region=region)
        df.append(['model',str(region),aname,'RMSE Score','score',s.units(),float(s.ds[s.varname].values)])
    df = pd.DataFrame(df,columns=['Model','Region','Analysis','ScalarName','ScalarType','Units','Data'])        
    r_plot,c_plot = add_analysis_name(aname,r_plot,c_plot)
    return r_plot,c_plot,df

def ScoreCycle(r0,c0,r=None,c=None,regions=[None]):
    """

    """
    if (r0.ds['time'].size < 12 or c0.ds['time'].size < 12): return {},{},[]
    v = r0.varname
    aname = "Annual Cycle"
    sdim = "site" if r0.sites() else "space"
    
    # compute cycle and month of maximum
    rc0 = r0 if r0.ds['time'].size==12 else r0.cycle()
    cc0 = c0 if c0.ds['time'].size==12 else c0.cycle()
    rmx0 = rc0.maxMonth()
    cmx0 = cc0.maxMonth()

    # phase shift on nested grid
    rmx,cmx = rmx0.nestSpatialGrids(cmx0)
    ps = cmx-rmx
    
    # manually fix the phase shift to [-6,+6]
    attrs = ps.ds[ps.varname].attrs
    ps.ds[ps.varname] = xr.where(ps.ds[ps.varname]<-6,ps.ds[ps.varname]+12,ps.ds[ps.varname])
    ps.ds[ps.varname] = xr.where(ps.ds[ps.varname]>+6,ps.ds[ps.varname]-12,ps.ds[ps.varname])
    ps.ds[ps.varname].attrs = attrs
    
    # compute score
    score = Variable(da = xr.apply_ufunc(lambda a: 1-np.abs(a)/6,ps.ds[ps.varname]),
                     varname = "shiftscore_map_of_%s" % v,
                     cell_measure = ps.ds['cell_measure'] if 'cell_measure' in ps.ds else None)
    score.ds[score.varname].attrs['units'] ='1'
    
    # collect output for intermediate files
    r_plot = {
        'phase_map_of_%s' % v: rmx0
    }
    c_plot = {
        'phase_map_of_%s' % v: cmx0,
        'shift_map_of_%s' % v: ps,
        'shiftscore_map_of_%s' % v: score
    }
    df = []
    for region in regions:
        r_plot['cycle_of_%s_over_%s' % (v,region)] = rc0.integrate(sdim,mean=True,region=region)
        c_plot['cycle_of_%s_over_%s' % (v,region)] = cc0.integrate(sdim,mean=True,region=region)
        s = ps.integrate(sdim,mean=True,region=region)
        df.append(['model',str(region),aname,'Phase Shift','scalar',s.units(),float(s.ds[s.varname].values)])
        s = score.integrate(sdim,mean=True,region=region)
        df.append(['model',str(region),aname,'Seasonal Cycle Score','score',s.units(),float(s.ds[s.varname].values)])
    df = pd.DataFrame(df,columns=['Model','Region','Analysis','ScalarName','ScalarType','Units','Data'])        
    r_plot,c_plot = add_analysis_name(aname,r_plot,c_plot)
    return r_plot,c_plot,df

def ScoreSpatialDistribution(r0,c0,r=None,c=None,regions=[None]):
    """

    """
    aname = "Spatial Distribution"
    if r0.temporal(): r0 = r0.integrate(dim='time',mean=True)
    if c0.temporal(): c0 = c0.integrate(dim='time',mean=True)
    r,c = pick_grid_aligned(r0,c0,r,c)
    df = []
    for region in regions:
        std0 = r.std(dim='space',region=region)
        std  = c.std(dim='space',region=region)/std0
        corr = r.correlation(c,'space',region=region)
        score = 2*(1+corr)/((std+1/std)**2)
        df.append(['model',str(region),aname,'Spatial Normalized Standard Deviation','scalar','1',std])
        df.append(['model',str(region),aname,'Spatial Correlation'                  ,'scalar','1',corr])
        df.append(['model',str(region),aname,'Spatial Distribution Score'           ,'score' ,'1',score])
    df = pd.DataFrame(df,columns=['Model','Region','Analysis','ScalarName','ScalarType','Units','Data'])        
    return {},{},df

class Confrontation(object):

    def __init__(self,**kwargs):
        """
        source
        variable
        unit
        regions
        master
        path
        cmap
        """
        self.source   = kwargs.get(  "source",None)
        self.variable = kwargs.get("variable",None)
        self.unit     = kwargs.get(    "unit",None)
        self.regions  = kwargs.get( "regions",[None])
        self.master   = kwargs.get(  "master",True)
        self.path     = kwargs.get(    "path","./")
        self.cmap     = kwargs.get(    "cmap",None)
        self.df_plot  = None
        assert self.source is not None
        if not os.path.isfile(self.source):
            msg = "Cannot find the source, tried looking here: '%s'" % self.source
            raise ValueError(msg)
        if not os.path.isdir(self.path): os.makedirs(self.path)
        
    def stageData(self,m):
        """
        
        """
        r = Variable(filename=self.source,varname=self.variable)
        if self.unit is not None: r.convert(self.unit)
        t0,tf = r.timeBounds()
        c = m.getVariable(self.variable,t0=t0,tf=tf)
        c.convert(r.units())
        r,c = adjust_lon(r,c)
        if r.sites() and c.spatial(): c = c.extractSites(r)
        return r,c
    
    def confront(self,m,**kwargs):
        """
        skip_bias
        skip_rmse
        skip_cycle
        skip_sd
        """
        # options
        skip_bias  = kwargs.get( 'skip_bias',False)
        skip_rmse  = kwargs.get( 'skip_rmse',False)
        skip_cycle = kwargs.get('skip_cycle',False)
        skip_sd    = kwargs.get(   'skip_sd',False)
                
        # initialize, detect what analyses are inappropriate
        r0,c0 = self.stageData(m)
        if not r0.temporal():
            skip_rmse  = True
            skip_cycle = True
        if not r0.spatial():
            skip_sd = True
        rplot = {}; cplot = {}; dfm = []

        # bias scoring
        if not skip_bias:
            rp,cp,df = ScoreBias(r0,c0,regions=self.regions)
            rplot.update(rp); cplot.update(cp); dfm.append(df)

        # rmse scoring
        if not skip_rmse:
            rp,cp,df = ScoreRMSE(r0,c0,regions=self.regions)
            rplot.update(rp); cplot.update(cp); dfm.append(df)
                
        # cycle scoring
        if not skip_cycle:
            rp,cp,df = ScoreCycle(r0,c0,regions=self.regions)
            rplot.update(rp); cplot.update(cp); dfm.append(df)

        # spatial distribution scoring
        if not skip_sd:
            name = 'timeint_of_%s' % self.variable
            ri = rplot[name] if name in rplot else r0
            ci = cplot[name] if name in cplot else c0
            rp,cp,df = ScoreSpatialDistribution(ri,ci,regions=self.regions)
            rplot.update(rp); cplot.update(cp); dfm.append(df)
        
        # compute overall score and output
        dfm = pd.concat([df for df in dfm if len(df)>0],ignore_index=True)
        dfm.Model[dfm.Model=='model'] = m.name
        dfm = overall_score(dfm)
        dfm.to_csv(os.path.join(self.path,"%s.csv" % m.name),index=False)

        # output maps and curves
        ds = sanitize_into_dataset(cplot)
        ds.attrs = {'name':m.name}
        ds.to_netcdf(os.path.join(self.path,"%s.nc" % m.name))
        if self.master:
            ds = sanitize_into_dataset(rplot)
            ds.attrs = {'name':'Reference'}
            ds.to_netcdf(os.path.join(self.path,"Reference.nc"))

    def _plot(self,m=None):
        """

        """
        name = "Reference" if m is None else m.name
        if self.df_plot is None:
            self.df_plot = generate_plot_database(glob.glob(os.path.join(self.path,"*.nc")),cmap=self.cmap)
        df = self.df_plot[(self.df_plot.Model==name) & (self.df_plot.IsSpace==True)]
        for i,r in df.iterrows():
            v = Variable(filename=r['Filename'],varname=r['Variable'])
            for region in self.regions:
                v.plot(cmap=r['Colormap'],vmin=r['Plot Min'],vmax=r['Plot Max'],region=region,tight_layout=True)
                path = os.path.join(self.path,"%s_%s_%s.png" % (name,str(region),r['Variable'].split("_")[0]))
                plt.gcf().savefig(path)
                plt.close()
        
    def plotReference(self):
        """

        """
        self._plot()
        
    def plotModel(self,m):
        """

        """
        self._plot(m)
                
if __name__ == "__main__":
    from ModelResult import ModelResult
    import time
    
    M = [ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2"  ,name="CESM2"  ),
         ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5",name="CanESM5")]
    for m in M:
        m.findFiles()
        m.getGridInformation()

    C = [Confrontation(source = "/home/nate/data/ILAMB/DATA/gpp/FLUXNET2015/gpp.nc",
                       variable = "gpp",
                       unit = "g m-2 d-1",
                       regions = [None,"nhsa"],
                       path = "./_build/gpp/FLUXNET2015"),
         Confrontation(source = "/home/nate/data/ILAMB/DATA/gpp/FLUXCOM/tmp.nc",
                       variable = "gpp",
                       unit = "g m-2 d-1",
                       regions = [None,"nhsa"],
                       cmap = "Greens",
                       path = "./_build/gpp/FLUXCOM"),
         Confrontation(source = "/home/nate/work/ILAMB-Data/CLASS/pr.nc",
                       variable = "pr",
                       unit = "kg m-2 d-1",
                       path = "./_build/pr/CLASS/")]
    for c in C[1:2]:
        print(c.source,c.variable)
        for m in M:
            t0 = time.time()
            print("  %10s" % (m.name),end=' ',flush=True)
            #c.confront(m)
            dt = time.time()-t0
            print("%.0f" % dt)
        for m in M:
            t0 = time.time()
            print("  %10s" % (m.name),end=' ',flush=True)
            #c.plotModel(m)
            dt = time.time()-t0
            print("%.0f" % dt)
        c.plotReference()
