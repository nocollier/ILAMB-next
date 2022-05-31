import os
import glob
from Variable import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from Post import *
from Regions import Regions

ilamb_regions = Regions()

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

def make_time_comparable(a,b):
    """The time dimension must be in a comporable format.

    """
    if 'time' not in a.ds[a.varname].dims: return a,b
    if 'time' not in b.ds[b.varname].dims: return a,b
    for v in [a,b]:
        v.ds['time'] = pd.to_datetime(v.ds['time'].dt.strftime("%Y-%m-%d %H:%M"))
        if 'bounds' in v.ds['time'].attrs:
            tb = v.ds['time'].attrs['bounds']
            if tb in v.ds:
                v.ds[tb] = pd.to_datetime(v.ds[tb].dt.strftime("%Y-%m-%d %H:%M"))
    return a,b

def trim_time(a,b):
    """When comparing b to a, we need only the maximal amount of temporal overlap.

    """
    if 'time' not in a.ds[a.varname].dims: return a,b
    if 'time' not in b.ds[b.varname].dims: return a,b
    at0,atf = a.timeBounds()
    bt0,btf = b.timeBounds()
    tol = int(1*24*3600*1e9) # 1 day in nanoseconds
    t0 = max(at0,bt0)-tol
    tf = min(atf,btf)+tol
    a.ds = a.ds.sel(time=slice(t0,tf))
    b.ds = b.ds.sel(time=slice(t0,tf))
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
    for key in set(d.keys()).difference(set(ds.keys())): ds[key] = d[key].ds[key]
    keep = ['_FillValue','units','ilamb','analysis','region','longname','rendered']
    for name in ds:
        ds[name].attrs = { key:val for key,val in ds[name].attrs.items() if key in keep }
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

def ScoreBias(r0,c0,r=None,c=None,regions=[None],df_errs=None):
    """

    """
    v = r0.varname
    aname = "Bias"
    sdim = "site" if r0.sites() else "space"

    # period means on original grids
    rm0 = r0.integrate(dim='time',mean=True) if r0.temporal() else r0
    cm0 = c0.integrate(dim='time',mean=True) if c0.temporal() else c0
    rm0.setAttr("longname","Temporal Mean")
    cm0.setAttr("longname","Temporal Mean")

    if df_errs is None: # as in (Collier, et al., JAMES, 2018)

        # if we have temporal data, the normalizer is the std
        norm0 = r0.std(dim='time') if r0.ds['time'].size > 1 else rm0

        # interpolate to a nested grid
        rm,cm,norm = rm0.nestSpatialGrids(cm0,norm0)
        bias = cm-rm
        bias.setAttr("longname","Bias")

        # do we have reference uncertainties?
        ru = rm.uncertainty()
        un = 0 if ru is None else ru.ds[ru.varname]

        # compute the bias score
        eps  = cm-rm
        eps.ds[eps.varname] = (np.abs(eps.ds[eps.varname])-un).clip(0)
        eps.ds[eps.varname].attrs['units'] = cm.units()
        eps /= norm
        eps.ds[eps.varname] = np.exp(-eps.ds[eps.varname])
        eps.setAttr("longname","Bias Score")

    else: # new methodology based on bias quantiles

        # interpolate to a nested grid
        rm,cm = rm0.nestSpatialGrids(cm0)
        bias = cm-rm
        bias.setAttr("longname","Bias")

        # do we have reference uncertainties?
        ru = rm.uncertainty()
        un = 0 if ru is None else ru.ds[ru.varname]

        # build up the score
        eps = cm-rm
        eps.ds[eps.varname] = (np.abs(eps.ds[eps.varname])-un).clip(0)
        for region in df_errs.region.unique():
            mask = ilamb_regions.getMask(region,bias)
            val  = float(df_errs.loc[(df_errs.variable   == v) &
                                     (df_errs.region     == region) &
                                     (df_errs.percentile == 98),'bias'])
            eps.ds[eps.varname] /= xr.where(mask,1,val)
        eps.ds[eps.varname] = (1-np.abs(eps.ds[eps.varname])).clip(0,1)
        eps.setAttr("units","1")
        eps.setAttr("longname","Bias Score")

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

def ScoreRMSE(r0,c0,r=None,c=None,regions=[None],df_errs=None):
    """

    """
    v = r0.varname
    aname = "RMSE"
    sdim = "site" if r0.sites() else "space"

    # validity checks
    if r0.ds['time'].size < 12: return {},{},[]

    if df_errs is None: # as in (Collier, et al., JAMES, 2018)

        # get normalizer and regrid
        norm0 = r0.std(dim='time')
        r,c,norm = r0.nestSpatialGrids(c0,norm0)

        # do we have reference uncertainties?
        ru = r.uncertainty()
        un = 0 if ru is None else ru.ds[ru.varname]

        # compute the RMSE error and score
        rmse = r.rmse(c)
        rmse.setAttr("longname","RMSE")
        r = r.detrend(degree=0)
        c = c.detrend(degree=0)
        crmse = r.rmse(c)
        crmse.setAttr("longname","Centralized RMSE")
        eps = c-r
        del c,r
        eps.ds[eps.varname] = (np.abs(eps.ds[eps.varname])-un).clip(0)**2
        eps.ds[eps.varname].attrs['units'] = "1"
        eps = eps.integrate(dim='time',mean=True)
        eps /= norm
        eps.ds[eps.varname] = np.exp(-eps.ds[eps.varname])
        eps.ds[eps.varname].attrs['units'] = "1"
        eps.setAttr("longname","RMSE Score")

    else:

        # interpolate to a nested grid
        r,c = r0.nestSpatialGrids(c0)

        # do we have reference uncertainties?
        ru = r.uncertainty()
        un = 0 if ru is None else ru.ds[ru.varname]

        # compute the RMSE error and score
        rmse = r.rmse(c)
        rmse.setAttr("longname","RMSE")
        r = r.detrend(degree=0)
        c = c.detrend(degree=0)
        crmse = r.rmse(c)
        crmse.setAttr("longname","Centralized RMSE")

        # build up the score
        eps = r.rmse(c,uncertainty=un)
        del c,r
        for region in df_errs.region.unique():
            mask = ilamb_regions.getMask(region,crmse)
            val  = float(df_errs.loc[(df_errs.variable   == v) &
                                     (df_errs.region     == region) &
                                     (df_errs.percentile == 98),'crmse'])
            eps.ds[eps.varname] /= xr.where(mask,1,val)
        eps.ds[eps.varname] = (1-np.abs(eps.ds[eps.varname])).clip(0,1)
        eps.setAttr("units","1")
        eps.setAttr("longname","RMSE Score")

    # collect output for intermediate files
    r_plot = {}
    c_plot = {
        'rmse_map_of_%s'      % v: rmse,
        'crmse_map_of_%s'     % v: crmse,
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
    rmx0.setAttr("longname","Month of Maximum")
    cmx0.setAttr("longname","Month of Maximum")

    # phase shift on nested grid
    rmx,cmx = rmx0.nestSpatialGrids(cmx0)
    ps = cmx-rmx

    # manually fix the phase shift to [-6,+6]
    attrs = ps.ds[ps.varname].attrs
    ps.ds[ps.varname] = xr.where(ps.ds[ps.varname]<-6,ps.ds[ps.varname]+12,ps.ds[ps.varname])
    ps.ds[ps.varname] = xr.where(ps.ds[ps.varname]>+6,ps.ds[ps.varname]-12,ps.ds[ps.varname])
    ps.ds[ps.varname].attrs = attrs
    ps.setAttr("longname","Phase Shift")

    # compute score
    score = Variable(da = xr.apply_ufunc(lambda a: 1-np.abs(a)/6,ps.ds[ps.varname]),
                     varname = "shiftscore_map_of_%s" % v,
                     cell_measure = ps.ds['cell_measure'] if 'cell_measure' in ps.ds else None)
    score.ds[score.varname].attrs['units'] ='1'
    score.setAttr("longname","Seasonal Cycle Score")

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
    v = r0.varname.split("_")[0]
    if r0.temporal(): r0 = r0.integrate(dim='time',mean=True)
    if c0.temporal(): c0 = c0.integrate(dim='time',mean=True)
    r,c = pick_grid_aligned(r0,c0,r,c)
    df = []; r_plot = {}
    for region in regions:
        std0 = r.std(dim='space',region=region)
        std  = c.std(dim='space',region=region)
        corr = r.correlation(c,'space',region=region)
        df.append(['Reference',str(region),aname,'Spatial Standard Deviation','scalar',r.units(),std0])
        df.append([    'model',str(region),aname,'Spatial Standard Deviation','scalar',r.units(),std])
        df.append([    'model',str(region),aname,'Spatial Correlation'       ,'scalar','1',corr])
        std  /= std0
        score = 2*(1+corr)/((std+1/std)**2)
        df.append([    'model',str(region),aname,'Spatial Distribution Score','score' ,'1',score])

    # Note: Plots and html pages are generated automatically from the
    # data in the netCDF files. In this case, we have made a custom
    # plot here. But we still need to add an entry that we will flag
    # as 'rendered'. This will get the figure added to the html page
    # in the right section.
    d = r0.integrate(dim='space',mean=True)
    d.setAttr("rendered",1)
    d.setAttr("longname","Spatial Distribution")
    n = "sd_of_%s" % (v)
    d.ds = d.ds.rename({d.varname:n})
    d.varname = n
    r_plot[n] = d
    r_plot,_ = add_analysis_name(aname,r_plot,{})
    df = pd.DataFrame(df,columns=['Model','Region','Analysis','ScalarName','ScalarType','Units','Data'])
    return r_plot,{},df

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
        self.df_errs  = kwargs.get( "df_errs",None)
        self.df_plot  = None
        self.df_scalar = None
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
        if type(t0) != str: t0 = str(t0.values)
        if type(tf) != str: tf = str(tf.values)
        c = m.getVariable(self.variable,t0=t0,tf=tf)
        c.convert(r.units())
        r,c = make_time_comparable(r,c)
        r,c = trim_time(r,c)
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
            rp,cp,df = ScoreBias(r0,c0,regions=self.regions,df_errs=self.df_errs)
            rplot.update(rp); cplot.update(cp); dfm.append(df)

        # rmse scoring
        if not skip_rmse:
            rp,cp,df = ScoreRMSE(r0,c0,regions=self.regions,df_errs=self.df_errs)
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
        dfm.loc[dfm.Model=='model','Model'] = m.name
        dfm = overall_score(dfm)
        dfm.to_csv(os.path.join(self.path,"%s.csv" % m.name),index=False)

        # output maps and curves
        ds = sanitize_into_dataset(cplot)
        ds.attrs = {'name':m.name,'color':m.color}
        ds.to_netcdf(os.path.join(self.path,"%s.nc" % m.name))
        if self.master:
            ds = sanitize_into_dataset(rplot)
            ds.attrs = {'name':'Reference','color':(0,0,0)}
            ds.to_netcdf(os.path.join(self.path,"Reference.nc"))

    def _plot(self,m=None):
        """

        """
        name = "Reference" if m is None else m.name
        if self.df_plot is None:
            self.df_plot = generate_plot_database(glob.glob(os.path.join(self.path,"*.nc")),cmap=self.cmap)

        # Map plots
        df = self.df_plot[(self.df_plot.Model==name) & ( (self.df_plot.IsSpace==True) |
                                                         (self.df_plot.IsSite ==True) )]
        for i,r in df.iterrows():
            v = Variable(filename=r['Filename'],varname=r['Variable'])
            for region in self.regions:
                v.plot(cmap=r['Colormap'],vmin=r['Plot Min'],vmax=r['Plot Max'],region=region,tight_layout=True)
                path = os.path.join(self.path,"%s_%s_%s.png" % (name,str(region),r['Variable'].split("_")[0]))
                plt.gca().set_title(name + " " + r['Longname'])
                plt.gcf().savefig(path)
                plt.close()

    def plotReference(self):
        """

        """
        if self.master: self._plot()

    def plotModel(self,m):
        """

        """
        self._plot(m)

    def plotComposite(self,M):
        """

        """
        if not self.master: return
        if self.df_scalar is None:
            self.df_scalar = generate_scalar_database(glob.glob(os.path.join(self.path,"*.csv")))
        dfs = self.df_scalar

        # Spatial distribution Taylor plot
        clrs = { m.name: m.color for m in M }
        if 'Spatial Distribution' in dfs['Analysis'].unique():
            df = dfs[dfs['Analysis']=='Spatial Distribution']
            for region in df['Region'].unique():
                dfr = df[df['Region']==region]
                r = dfr[(dfr['Model']=='Reference')]
                if len(r) != 1: continue
                r = r.iloc[0]

                # determine range of standard deivation in Taylor plot
                rng = dfr.loc[dfr['ScalarName']=='Spatial Standard Deviation','Data']/r['Data']
                pad = 0.1*(rng.max()-rng.min())
                rng = rng.append(pd.Series([rng.max()+pad,1.5,0.5,rng.min()-pad]))

                fig,[ax0,ax1] = plt.subplots(ncols=2,figsize=(6,5),dpi=200,gridspec_kw={'width_ratios': [4, 1]},tight_layout=True)
                td = TaylorDiagram(r['Data'],fig=fig,rect=121,label="Reference",srange=(rng.min(),rng.max()))
                for model in dfr['Model'].unique():
                    c = dfr[(dfr['Model']==model)]
                    if len(c) != 3: continue
                    td.add_sample(c[c['ScalarName']=='Spatial Standard Deviation']['Data'],
                                  c[c['ScalarName']==       'Spatial Correlation']['Data'],
                                  marker='o', ms=7, ls='', mfc=clrs[model], mec=clrs[model], label=model)
                td.add_grid()
                contours = td.add_contours(colors='0.8')
                plt.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
                ax1.legend(td.samplePoints,
                           [ p.get_label() for p in td.samplePoints ],
                           numpoints=1, prop=dict(size='small'), loc='upper right')
                ax0.axis('off'); ax1.axis('off')
                fig.suptitle("Spatial Distribution")
                path = os.path.join(self.path,"Reference_%s_sd.png" % (str(region)))
                fig.savefig(path)
                plt.close()

    def generateHTML(self):
        """

        """
        if not self.master: return
        if self.df_plot is None:
            self.df_plot = generate_plot_database(glob.glob(os.path.join(self.path,"*.nc")),cmap=self.cmap)
        if self.df_scalar is None:
            self.df_scalar = generate_scalar_database(glob.glob(os.path.join(self.path,"*.csv")))
        dfp = self.df_plot
        dfs = self.df_scalar
        html = generate_dataset_html(dfp,dfs,self.source,self.variable)
        with open(os.path.join(self.path,"index.html"),mode='w') as f:
            f.write(html)


def assign_model_colors(M):
    """Later migrate this elsewhere.

    """
    n = len(M)
    if n <= 10:
        clrs = np.asarray(plt.get_cmap("tab10").colors)
    elif n <= 20:
        clrs = np.asarray(plt.get_cmap("tab20").colors)
    elif n <= 40:
        clrs = np.vstack([plt.get_cmap("tab20b").colors,plt.get_cmap("tab20c").colors])
    else:
        msg = "We only have 40 colors to assign to models"
        raise ValueError(msg)
    for i,m in enumerate(M):
        m.color = clrs[i]

if __name__ == "__main__":
    from ModelResult import ModelResult
    from Regions import Regions
    import time

    ROOT = os.environ['ILAMB_ROOT']
    Regions().addRegionNetCDF4(os.path.join(ROOT,"DATA/regions/Whittaker.nc"))

    M = [
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/bcc-csm1-1"   ),name="bcc-csm1-1"   ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/CanESM2"      ),name="CanESM2"      ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/CESM1-BGC"    ),name="CESM1-BGC"    ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/GFDL-ESM2G"   ),name="GFDL-ESM2G"   ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/IPSL-CM5A-LR" ),name="IPSL-CM5A-LR" ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/MIROC-ESM"    ),name="MIROC-ESM"    ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/MPI-ESM-LR"   ),name="MPI-ESM-LR"   ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/NorESM1-ME"   ),name="NorESM1-ME"   ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP5/HadGEM2-ES"   ),name="UK-HadGEM2-ES"),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/BCC-CSM2-MR"  ),name="BCC-CSM2-MR"  ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/CanESM5"      ),name="CanESM5"      ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/CESM2"        ),name="CESM2"        ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/GFDL-ESM4"    ),name="GFDL-ESM4"    ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/IPSL-CM6A-LR" ),name="IPSL-CM6A-LR" ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/MIROC-ES2L"   ),name="MIROC-ES2L"   ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/MPI-ESM1.2-HR"),name="MPI-ESM1.2-HR"),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/NorESM2-LM"   ),name="NormESM2-LM"  ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/UKESM1-0-LL"  ),name="UKESM1-0-LL"  ),
    ]
    M = [ModelResult(os.path.join(ROOT,"MODELS/CMIP6/CESM2"        ),name="CESM2"        )]
    print("Initialize models...")
    for m in M:
        m.findFiles()
        m.getGridInformation()
        print("  ",m.name)
    assign_model_colors(M)

    df = pd.read_pickle('cmip5v6_errors.pkl')
    C = [Confrontation(source = os.path.join(ROOT,"DATA/gpp/FLUXNET2015/gpp.nc"),
                       variable = "gpp",
                       unit = "g m-2 d-1",
                       regions = [None,"euro"],
                       cmap = "Greens",
                       path = "./_build/gpp/FLUXNET2015",
                       df_errs = df),
         Confrontation(source = os.path.join(ROOT,"DATA/gpp/FLUXCOM/gpp.nc"),
                       variable = "gpp",
                       unit = "g m-2 d-1",
                       regions = [None,"euro"],
                       cmap = "Greens",
                       path = "./_build/gpp/FLUXCOM",
                       df_errs = df)]
    for c in C:
        print(c.source,c.variable)
        for m in M:
            t0 = time.time()
            print("  %20s" % (m.name),end=' ',flush=True)
            try:
                c.confront(m,skip_bias=False,skip_rmse=False,skip_cycle=False,skip_sd=False)
                dt = time.time()-t0
                print("%.0f" % dt)
            except:
                print("fail")
        for m in M:
            t0 = time.time()
            print("  %20s" % (m.name),end=' ',flush=True)
            c.plotModel(m)
            dt = time.time()-t0
            print("%.0f" % dt)
        c.plotComposite(M)
        c.plotReference()
        c.generateHTML()
