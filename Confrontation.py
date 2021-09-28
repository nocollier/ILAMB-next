import os
from Variable import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

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

    * make sure the dataarray names match the keys of the dictionary
    * add '_' to time/lat/lon dimensions if more than one unique set exists
    * weed out attributes which were carried over from the source files

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
    keep = ['_FillValue','units','ilamb']
    for name in ds: ds[name].attrs = { key:val for key,val in ds[name].attrs.items() if key in keep }
    ds = xr.Dataset(ds)
    return ds
    
def ScoreBias(r0,c0,r=None,c=None,model_name="",regions=[None]):
    """

    """
    v = r0.varname
    
    # period means on original grids
    rm0 = r0.integrateInTime(mean=True)
    cm0 = c0.integrateInTime(mean=True)

    # if we have temporal data, the normalizer is the std
    norm0 = r0.stddev() if r0.ds['time'].size > 1 else rm0

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
        s = rm0.integrateInSpace(mean=True,region=region)
        df.append(['Benchmark',region,'Period Mean','scalar',s.units(),float(s.ds[s.varname].values)])
        s = cm0.integrateInSpace(mean=True,region=region)
        df.append([model_name,region,'Period Mean','scalar',s.units(),float(s.ds[s.varname].values)])
        s = bias.integrateInSpace(mean=True,region=region)
        df.append([model_name,region,'Bias','scalar',s.units(),float(s.ds[s.varname].values)])
        s = eps.integrateInSpace(mean=True,region=region)
        df.append([model_name,region,'Bias Score','score',s.units(),float(s.ds[s.varname].values)])
    df = pd.DataFrame(df,columns=['Model','Region','ScalarName','ScalarType','Units','Data'])

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
    return r_plot,c_plot,df

def ScoreCycle(r0,c0,r=None,c=None,model_name="",regions=[None]):
    """

    """
    if (r0.ds['time'].size < 12 or c0.ds['time'].size < 12): return {},{},[]
    v = r0.varname
    
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
        'phase_map_of_%s' % v: cmx0
    }
    df = []
    for region in regions:
        r_plot['cycle_of_%s_over_%s' % (v,region)] = rc0.integrateInSpace(mean=True,region=region)
        c_plot['cycle_of_%s_over_%s' % (v,region)] = cc0.integrateInSpace(mean=True,region=region)
        s = ps.integrateInSpace(mean=True,region=region)
        df.append([model_name,region,'Phase Shift','scalar',s.units(),float(s.ds[s.varname].values)])
        s = score.integrateInSpace(mean=True,region=region)
        df.append([model_name,region,'Seasonal Cycle Score','score',s.units(),float(s.ds[s.varname].values)])
    df = pd.DataFrame(df,columns=['Model','Region','ScalarName','ScalarType','Units','Data'])        
    return r_plot,c_plot,df
    
class Confrontation(object):

    def __init__(self,**kwargs):
        """
        source
        variable
        unit
        regions
        master
        path
        """
        self.source   = kwargs.get(  "source",None)
        self.variable = kwargs.get("variable",None)
        self.unit     = kwargs.get(    "unit",None)
        self.regions  = kwargs.get( "regions",[None])
        self.master   = kwargs.get(  "master",True)
        self.path     = kwargs.get(    "path","./")
        assert self.source is not None
        if not os.path.isfile(self.source):
            msg = "Cannot find the source, tried looking here: '%s'" % self.source
            raise ValueError(msg)
        if not os.path.isdir(self.path): os.makedirs(self.path)
        
    def stageData(self,m):
        """
        
        """
        r = Variable(filename = self.source,
                     varname  = self.variable)
        if self.unit is not None: r.convert(self.unit)
        t0,tf = r.timeBounds()
        c = m.getVariable(self.variable,t0=t0,tf=tf)
        c.convert(r.units())
        r,c = adjust_lon(r,c)
        return r,c
    
    def confront(self,m):
        """

        """
        r0,c0 = self.stageData(m)

        # initialize output containers
        rplot = {}; cplot = {}; dfm = []

        # bias scoring
        rp,cp,df = ScoreBias(r0,c0,model_name=m.name,regions=self.regions)
        rplot.update(rp); cplot.update(cp); dfm.append(df)

        # rmse scoring
        
        # cycle scoring
        rp,cp,df = ScoreCycle(r0,c0,model_name=m.name,regions=self.regions)
        rplot.update(rp); cplot.update(cp); dfm.append(df)

        # spatial distribution scoring
        
        # write outputs
        dfm = pd.concat([df for df in dfm if len(df)>0]).reset_index(drop=True)
        dfm.to_csv(os.path.join(self.path,"%s.csv" % m.name))        
        ds = sanitize_into_dataset(cplot)
        ds.to_netcdf(os.path.join(self.path,"%s.nc" % m.name))
        if self.master:
            ds = sanitize_into_dataset(rplot)
            ds.to_netcdf(os.path.join(self.path,"Benchmark.nc"))

        return
        
        
if __name__ == "__main__":
    from ModelResult import ModelResult

    M = [ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2",name="CESM2"),
         ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5",name="CanESM5")]
    for m in M:
        m.findFiles()
        m.getGridInformation()

    c = Confrontation(source = "/home/nate/data/ILAMB/DATA/gpp/FLUXCOM/tmp.nc",
                      variable = "gpp",
                      unit = "g m-2 d-1",
                      regions = [None,"nhsa"],
                      path = "./_build/gpp/FLUXCOM")
    for m in M: c.confront(m)

    if 1:
        c = Confrontation(source = "/home/nate/work/ILAMB-Data/CLASS/pr.nc",
                          variable = "pr",
                          unit = "kg m-2 d-1",
                          path = "./_build/pr/CLASS/")
        for m in M: c.confront(m)
    

