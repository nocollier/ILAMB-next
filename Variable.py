import xarray as xr
from cf_units import Unit
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from Regions import Regions

ilamb_regions = Regions()

def align_latlon(a,b):
    """We need some conditional checks on aligning.

    FIX: Need to check all dimensions, not just these
    """
    lata = a.ds[a.lat_name]; latb = b.ds[b.lat_name]
    lona = a.ds[a.lon_name]; lonb = b.ds[b.lon_name]
    if np.allclose(lata,latb,atol=1e-3) and np.allclose(lona,lonb,atol=1e-3):
        a.ds,b.ds = xr.align(a.ds,b.ds,join='override',copy=False)
        return a,b
    raise ValueError("Could not align")

def extents_space(da,lat_name,lon_name):
    """Find the extent of the non-null data.

    """
    nl = (da.isnull()==False)
    lat = np.where(nl.sum(lon_name)>0)[0]
    lon = np.where(nl.sum(lat_name)>0)[0]
    return np.asarray([float(da[lon_name][lon[0]]),
                       float(da[lon_name][lon[-1]]),
                       float(da[lat_name][lat[0]]),
                       float(da[lat_name][lat[-1]])])

def extents_sites(da,lat,lon):
    """Find the extent of the non-null data.

    """
    nl = (da.isnull()==False)
    sdim = find_site_dimension(da)
    dims = set(da.dims).difference(sdim)
    if dims: nl = nl.any(dims)
    ext = [float(lon[nl].min()),float(lon[nl].max()),float(lat[nl].min()),float(lat[nl].max())]
    return np.asarray(ext)

def pick_projection(extents):
    """Given plot extents choose projection.

    """
    if (extents[1]-extents[0]) > 300:
        aspect_ratio = 2.
        proj = ccrs.Robinson(central_longitude=0)
        if (extents[2] >   0) and (extents[3] > 75):
            aspect_ratio = 1.
            proj = ccrs.Orthographic(central_latitude=+90,central_longitude=0)
        if (extents[2] < -75) and (extents[3] <  0):
            aspect_ratio = 1.
            proj = ccrs.Orthographic(central_latitude=-90,central_longitude=0)
    else:
        aspect_ratio  = (max(extents[1],extents[0])-min(extents[1],extents[0]))
        aspect_ratio /= (max(extents[3],extents[2])-min(extents[3],extents[2]))
        proj = ccrs.PlateCarree(central_longitude=extents[:2].mean())
    return proj,aspect_ratio

def find_site_dimension(da):
    """

    """
    sdim = [d for d in da.dims if ((d not in da.coords) or np.issubdtype(da[d],np.integer))]
    return sdim

def move_coordinates(ds,varname):
    """Move coordinates from the dataset to the dataarray as appropriate.

    When reading in site data, the data site dimension is a scalar
    integer and usually there are one-dimensional arrays of the size
    of the number of data sites that contain lat/lon and other
    supplementary information. Find these and pass them as coordinates
    of the variable data array. This will enable logic internally to
    choose when a variable is spatial (lat/lon are dimensions) and
    when it is sites (lat/lon are coordinates only).

    """
    da = ds[varname]
    candidate_dims = [d for d in da.dims if d not in ds]
    coords = { key: ds[key] for key in ds if (ds[key].ndim==1 and
                                              ds[key].dims[0] in candidate_dims) }
    ds = ds.drop(coords.keys())
    ds[varname] = ds[varname].assign_coords(coords)
    return ds

class Variable():
    """Extends an xarray Dataset/Dataarray to track quantities we need to
       interpolation as well as provides implementations of
       integrals/means which are consistent with a functional
       interpretation of the input data.

    """
    def __init__(self,**kwargs):
        """
        varname
        filename
        da
        cell_measure
        time_measure
        """
        # Construction options, either 'filename' or 'da' must be specified
        self.ds = None
        self.varname  = kwargs.get( "varname","unnamed")
        self.filename = kwargs.get("filename",None)
        if self.filename is not None:
            self.ds = xr.open_dataset(self.filename)
            self.ds = move_coordinates(self.ds,self.varname)
        else:
            da = kwargs.get("da",None)
            assert da is not None
            self.ds = xr.Dataset({self.varname:da})
            self.ds[self.varname].name = self.varname
            
        # Measures may be given in the constructor, if not then we build them
        if 'time' in self.ds[self.varname].dims:
            dt = kwargs.get("time_measure",None)
            if dt is None:
                self._createTimeMeasure()
            else:
                self.ds['time_measure'] = dt
        
        # Optionally trim away time to save memory
        t0 = kwargs.get("t0",None)
        tf = kwargs.get("tf",None)
        if 'time' in self.ds[self.varname].dims: self.ds = self.ds.sel(time=slice(t0,tf))
        
        # Is there a 'lat' or 'lon' dimension? If so, get the names
        da = self.ds[self.varname]
        self.lat_name = [d for d in da.dims if d.lower().startswith('lat')]
        self.lon_name = [d for d in da.dims if d.lower().startswith('lon')]
        if len(self.lat_name) > 1 or len(self.lon_name) > 1:
            msg  = "Ambiguity in which data array dimensions are spatial coordinates: "
            msg += "lat = ['%s'], lon = ['%s']" % (",".join(self.lat_name),",".join(self.lon_name))
            raise ValueError(msg)        
        self.lat_name = self.lat_name[0] if self.lat_name else None
        self.lon_name = self.lon_name[0] if self.lon_name else None

        # If not spatial, we still could have 'lat' and 'lon' in the coordinates
        if not self.spatial():
            self.lat_name = [d for d in da.coords if (d.lower().startswith('lat') and d not in da.dims)]
            self.lon_name = [d for d in da.coords if (d.lower().startswith('lon') and d not in da.dims)]
            if len(self.lat_name) > 1 or len(self.lon_name) > 1:
                msg  = "Ambiguity in which data array coordinates are spatial locations: "
                msg += "lat = ['%s'], lon = ['%s']" % (",".join(self.lat_name),",".join(self.lon_name))
                raise ValueError(msg)        
            self.lat_name = self.lat_name[0] if self.lat_name else None
            self.lon_name = self.lon_name[0] if self.lon_name else None
            
        # Read or build cell measures
        if self.spatial():
            self.is_regular = True
            dx = kwargs.get("cell_measure",None)
            if dx is None:
                self._createCellMeasure()
            else:
                self.ds,dx = xr.align(self.ds,dx,join='override',copy=False)
                self.ds['cell_measure'] = dx
        
    def __str__(self):
        da   = self.ds[self.varname]
        out  = da.__str__()
        out += "\nStatistics:"
        out += "\n    {0:<16}{1:.6e}".format("minimum:",da.min().data)
        out += "\n    {0:<16}{1:.6e}".format("mean:",da.mean().data)
        out += "\n    {0:<16}{1:.6e}".format("maximum:",da.max().data)
        return out

    def __repr__(self):
        return self.__str__()

    def __sub__(self,other):
        other.convert(self.units())
        self,other = align_latlon(self,other)
        cm =  self.ds['cell_measure'] if  'cell_measure' in  self.ds                 else None
        cm = other.ds['cell_measure'] if ('cell_measure' in other.ds and cm is None) else None
        tm =  self.ds['time_measure'] if  'time_measure' in  self.ds                 else None
        tm = other.ds['time_measure'] if ('time_measure' in other.ds and cm is None) else None
        out = self.ds[self.varname]-other.ds[other.varname]
        out.name = "diff"
        out.attrs['units'] = self.units()
        return Variable(da = out, varname = "diff", cell_measure = cm, time_measure = tm)

    def __truediv__(self,other):
        self,other = align_latlon(self,other)
        cm =  self.ds['cell_measure'] if  'cell_measure' in  self.ds                 else None
        cm = other.ds['cell_measure'] if ('cell_measure' in other.ds and cm is None) else None
        tm =  self.ds['time_measure'] if  'time_measure' in  self.ds                 else None
        tm = other.ds['time_measure'] if ('time_measure' in other.ds and cm is None) else None
        out = Variable(da = self.ds[self.varname]/other.ds[other.varname],
                       cell_measure = cm, time_measure = tm)
        u = Unit( self.units())/Unit(other.units())
        out.ds[out.varname].attrs['units'] = str(u)
        return out

    def temporal(self):
        if 'time' in self.ds[self.varname].dims: return True
        return False

    def spatial(self):
        if self.lat_name is None: return False
        if self.lon_name is None: return False
        if self.lat_name not in self.ds[self.varname].dims: return False
        if self.lon_name not in self.ds[self.varname].dims: return False
        return True

    def sites(self):
        if self.spatial(): return False
        sdim = find_site_dimension(self.ds[self.varname])
        if len(sdim) > 1:
            msg = "Ambiguous site dimensions found [%s]" % (",".join([str(s) for s in sdim]))
            raise ValueError(msg)
        if sdim: return True
    
    def timeBounds(self):
        """Return the time extent of the dataset/array
        """
        if "time" not in self.ds[self.varname].dims: return None
        t = self.ds["time"]
        if 'bounds' in t.attrs:
            if t.attrs['bounds'] in self.ds:
                t = self.ds[t.attrs['bounds']]
        return t.min(),t.max()

    def rmse(self,other):
        """

        """
        out = other-self
        out.ds[out.varname] *= out.ds[out.varname]
        out = out.integrate(dim='time',mean=True)
        out.ds[out.varname] = np.sqrt(out.ds[out.varname])
        return out
        
    def uncertainty(self):
        """

        """
        if "bounds" not in self.ds[self.varname].attrs: return None
        if self.ds[self.varname].attrs['bounds'] not in self.ds: return None
        # FIX: need to detect the 'nb' dimension
        up  = self.ds[self.ds[self.varname].attrs['bounds']].isel({'nb':1})
        dn  = self.ds[self.ds[self.varname].attrs['bounds']].isel({'nb':0})
        mid = self.ds[self.varname]
        u   = np.sqrt((up-mid)**2+(dn-mid)**2)
        u.attrs['units'] = self.units()
        cm  = self.ds['cell_measure'] if 'cell_measure' in self.ds else None
        tm  = self.ds['time_measure'] if 'time_measure' in self.ds else None
        return Variable(da = u,cell_measure = cm, time_measure = tm)
        
    def units(self):
        if "units" not in self.ds[self.varname].attrs: return "1"
        return self.ds[self.varname].attrs['units']

    def setAttr(self,attr,value):
        self.ds[self.varname].attrs[attr] = value
    
    def convert(self,unit,density=998.2,molar_mass=12.011):
        """Using cf_units (UDUNITS2) convert the unit in place
        - handles ( M L-2 T-1 ) --> ( L T-1 ), assuming water
        - handles (       mol ) --> (     M ), assuming carbon
        """
        if 'units' not in self.ds[self.varname].attrs:
            msg = "Cannot convert the units of the DataArray lacking the 'units' attribute"
            raise ValueError(msg)
        src_unit = Unit(self.ds[self.varname].units)
        tar_unit = Unit(unit)
        mass_density = Unit("kg m-3")
        molar_density = Unit("g mol-1")
        if ((src_unit/tar_unit)/mass_density).is_dimensionless():
            self.ds[self.varname] /= density
            src_unit /= mass_density
        elif ((tar_unit/src_unit)/mass_density).is_dimensionless():
            self.ds[self.varname] *= density
            src_unit *= mass_density
        if ((src_unit/tar_unit)/molar_density).is_dimensionless():
            self.ds[self.varname] /= molar_mass
            src_unit /= molar_density
        elif ((tar_unit/src_unit)/molar_density).is_dimensionless():
            self.ds[self.varname] *= molar_mass
            src_unit *= molar_density
        src_unit.convert(self.ds[self.varname].data,tar_unit,inplace=True)
        self.ds[self.varname].attrs['units'] = unit
        if "bounds" in self.ds[self.varname].attrs:
            if self.ds[self.varname].attrs['bounds'] in self.ds:                
                src_unit.convert(self.ds[self.ds[self.varname].attrs['bounds']].data,tar_unit,inplace=True)
        return self

    def extractSites(self,v=None,lat=None,lon=None):
        """
        """   
        assert self.spatial()
        if v is not None:
            da = self.ds[self.varname]
            lat_name = [c for c in da.coords if c.lower().startswith("lat")]
            lon_name = [c for c in da.coords if c.lower().startswith("lon")]
            if lat_name and lon_name:
                lat = v.ds[lat_name[0]]
                lon = v.ds[lon_name[0]]
        assert lat is not None
        assert lon is not None
        tm = self.ds.time_measure if "time_measure" in self.ds else None
        coords = {'lat':lat,'lon':lon}
        da = self.ds[self.varname].sel(coords,method='nearest').assign_coords(coords)
        v = Variable(da = da, varname = str(da.name), time_measure = tm)
        return v

    def plot(self,**kwargs):
        """
        """
        def _finalize_plot(ax,ext):
            ax.add_feature(cfeature.NaturalEarthFeature('physical','land','110m',
                                                        edgecolor='face',
                                                        facecolor='0.875'),zorder=-1)
            ax.add_feature(cfeature.NaturalEarthFeature('physical','ocean','110m',
                                                        edgecolor='face',
                                                        facecolor='0.750'),zorder=-1)
            # cleanup plotting extents
            percent_pad = 0.1
            if (ext[1]-ext[0]) > 330:
                ext[:2] = [-180,180] # set_extent doesn't like (0,360)
                ext[2:] = [- 90, 90]
            else:
                dx = percent_pad*(ext[1]-ext[0])
                dy = percent_pad*(ext[3]-ext[2])
                ext[0] -= dx; ext[1] += dx
                ext[2] -= dy; ext[3] += dy
            ax.set_extent(ext,ccrs.PlateCarree())
            return ax
        
        region = kwargs.get("region",None)
        if "region" in kwargs: kwargs.pop("region")
        if "cmap"   in kwargs: kwargs['cmap'] = plt.get_cmap(kwargs['cmap'],9)
        
        ds = self.ds
        if region is not None: ds = ilamb_regions.getMask(region,self)
        da = ds[self.varname]
        if not self.temporal() and self.spatial():
            if "cell_measure" in ds: da = xr.where(ds['cell_measure']<1,np.nan,da)
            ext = extents_space(da,self.lat_name,self.lon_name)
            proj,aspect = pick_projection(ext)
            figsize = kwargs.pop('figsize') if 'figsize' in kwargs else (6*1.03,6/aspect)
            fig,ax = plt.subplots(dpi=200,
                                  tight_layout=kwargs.pop('tight_layout') if 'tight_layout' in kwargs else None,
                                  figsize=figsize,
                                  subplot_kw={'projection':proj})
            p = da.plot(ax          = ax,
                        transform   = ccrs.PlateCarree(),
                        cbar_kwargs = {'label':self.units()},
                        **kwargs)
            ax.set_title("")
            ax = _finalize_plot(ax,ext)
        elif not self.temporal() and self.sites():
            if self.lat_name is None or self.lon_name is None: return
            ext = extents_sites(da,self.ds[self.lat_name],self.ds[self.lon_name])
            proj,aspect = pick_projection(ext)
            fig,ax = plt.subplots(dpi=200,
                                  tight_layout=kwargs.pop('tight_layout') if 'tight_layout' in kwargs else None,
                                  figsize=kwargs.pop('figsize') if 'figsize' in kwargs else None,
                                  subplot_kw={'projection':proj})
            p = ax.scatter(self.ds[self.lon_name],self.ds[self.lat_name],c=self.ds[self.varname],
                           transform=ccrs.PlateCarree(),**kwargs)
            ax = _finalize_plot(ax,ext)
        else:
            da.plot(**kwargs)
            
    def _std_time(self):
        """Internal function for std in the time dimension.

        """
        mean = self.integrate(dim='time',mean=True)
        out = (self.ds[self.varname]-mean.ds[mean.varname])**2
        drops = ['time','actual_range']
        attrs = {key:a for key,a in self.ds[self.varname].attrs.items() if key not in drops}
        if 'ilamb' not in out.attrs: out.attrs['ilamb'] = ''
        attrs['ilamb'] += "std(dim='time'); "
        v = Variable(da = out, varname = 'none',
                     cell_measure = self.ds["cell_measure"] if "cell_measure" in self.ds else None,
                     time_measure = self.ds["time_measure"] if "time_measure" in self.ds else None)
        v.ds[v.varname].attrs = attrs        
        v = v.integrate(dim='time',mean=True)
        v.ds[v.varname] = np.sqrt(v.ds[v.varname])
        v.ds = v.ds.rename({v.varname:self.varname + "_std"})
        v.varname = self.varname + "_std"
        v.ds[v.varname].attrs = attrs        
        return v

    def _std_space(self,region=None):
        """Internal function for std in the space dimension.

        """
        mean = self.integrate(dim='space',region=region,mean=True)
        out = (self.ds[self.varname]-mean.ds[mean.varname])**2
        drops = ['actual_range']
        attrs = {key:a for key,a in self.ds[self.varname].attrs.items() if key not in drops}
        if 'ilamb' not in out.attrs: out.attrs['ilamb'] = ''
        attrs['ilamb'] += "std(dim='space'); "
        v = Variable(da = out, varname = 'none',
                     cell_measure = self.ds["cell_measure"] if "cell_measure" in self.ds else None,
                     time_measure = self.ds["time_measure"] if "time_measure" in self.ds else None)
        v.ds[v.varname].attrs = attrs        
        v = v.integrate(dim='space',region=region,mean=True)
        v.ds[v.varname] = np.sqrt(v.ds[v.varname])
        v.ds = v.ds.rename({v.varname:self.varname + "_std"})
        v.varname = self.varname + "_std"
        v.ds[v.varname].attrs = attrs        
        return v
    
    def std(self,dim,region=None):
        """
        """
        if dim ==  "time": v = self._std_time()
        if dim == "space": v = self._std_space(region)
        if v.ds[v.varname].size == 1: v = float(v.ds[v.varname])
        return v

    def _integrate_time(self,t0=None,tf=None,mean=False):
        """
        """
        da = self.ds[self.varname]
        if "time" not in da.dims:
            msg = "To integrateInTime you must have a dimension named 'time'"
            raise ValueError(msg)
        assert "time_measure" in self.ds
        ds = self.ds.sel(time=slice(t0,tf))
        da = self.ds[self.varname]
        dt = self.ds['time_measure']
        out = (da*dt).sum(dt.dims,min_count=1)
        out_bnds = None
        if "bounds" in da.attrs and da.attrs['bounds'] in self.ds:
            out_bnds = (self.ds[da.attrs['bounds']]*dt).sum(dt.dims,min_count=1)            
        units = Unit(da.units)
        attrs = {key:a for key,a in da.attrs.items() if ("time"         not in key and
                                                         "actual_range" not in key)}
        out.attrs = attrs
        if 'ilamb' not in out.attrs: out.attrs['ilamb'] = ''
        out.attrs['ilamb'] += "integrate(dim='time',t0='%s',tf='%s',mean=%s); " % (t0,tf,mean)
        if mean:
            out /= dt.sum()
            if out_bnds is not None: out_bnds /= dt.sum()
        else:
            units *= Unit("d")
            out.attrs['units'] = str(units)
        cm = self.ds["cell_measure"] if "cell_measure" in self.ds else None
        name = "none" if da.name is None else da.name
        v = Variable(da = out, varname = name + "_tint", cell_measure = cm)
        if out_bnds is not None:
            bnd_name = v.varname+"_bnds"
            v.ds[bnd_name] = out_bnds
            v.ds[v.varname].attrs['bounds'] = bnd_name
        return v
    
    def _integrate_space(self,region=None,mean=False):
        assert "cell_measure" in self.ds
        ds = self.ds if region is None else ilamb_regions.getMask(region,self)
        da = ds[self.varname]
        cm = ds['cell_measure']
        v,dx = xr.align(da,xr.where(cm<1,np.nan,cm),join='override',copy=False)
        out = (v*dx).sum(dx.dims)
        units = Unit(self.ds[self.varname].attrs['units'])
        out.attrs = {key:a for key,a in v.attrs.items() if "cell_" not in key}
        if 'ilamb' not in out.attrs: out.attrs['ilamb'] = ''
        out.attrs['ilamb'] += "integrate(dim='space',mean=%s%s); " % (mean,"" if region is None else ",region='%s'" % region)
        if mean:
            mask = da.isnull()
            dims = set(mask.dims).difference(set(dx.dims))
            if dims: mask = mask.all(dims)
            out /= (dx*(mask==False)).sum()
        else:
            if 'm-2' in str(units):
                units = str(units).replace("m-2","")
            else:
                units *= Unit('m2')
        out.attrs['units'] = str(units)
        tm = self.ds.time_measure if "time_measure" in self.ds else None
        v = Variable(da = out, varname = str(da.name) + "_sint", time_measure = tm)
        return v

    def _integrate_site(self,region=None,mean=False):
        """Integrate over sites, probably doesn't make sense but gives us a uniform API for taking means across space/sites.

        """
        ds = self.ds if region is None else ilamb_regions.getMask(region,self)
        da = ds[self.varname]
        sdim = find_site_dimension(da)
        if mean:
            out = da.mean(dim=sdim)
        else:
            out = da.sum(dim=sdim)            
        out.attrs['units'] = self.units()
        tm = self.ds.time_measure if "time_measure" in self.ds else None
        v = Variable(da = out, varname = str(da.name) + "_sint", time_measure = tm)
        return v
    
    def integrate(self,dim,region=None,mean=False,t0=None,tf=None):
        """
        """
        v = None
        assert dim in ['time','space','site']
        if dim == "space":
            assert self.spatial()
            v = self._integrate_space(region=region,mean=mean)
        if dim ==  "site":
            assert self.sites()
            v = self._integrate_site(region=region,mean=mean)
        if dim == "time":
            assert self.temporal()
            v = self._integrate_time(t0=t0,tf=tf,mean=mean)
        assert v is not None
        return v
        
    def detrend(self,dim=['time'],degree=1):
        """Remove a polynomial trend from each dimension, one at a time.

        """
        if type(dim) is not type([]): dim = [dim]
        assert set(dim).issubset(set(self.ds[self.varname].dims))
        for d in dim:
            p = self.ds[self.varname].polyfit(dim=d,deg=degree)
            fit = xr.polyval(self.ds[self.varname][d], p.polyfit_coefficients)
            self.ds[self.varname] -= fit            
        if 'ilamb' not in self.ds[self.varname].attrs: self.ds[self.varname].attrs['ilamb'] = ''
        dim = ["'%s'" % d for d in dim]
        self.ds[self.varname].attrs['ilamb'] += "detrend(dim=[%s],degree=%d); " % (",".join(dim),degree)
        return self
    
    def decycle(self):
        """Remove the annual cycle based on monthly means in place.

        """
        da = self.ds[self.varname]
        attrs = da.attrs
        gr = da.groupby('time.month')
        self.ds[self.varname] = gr - gr.mean(dim='time')
        if 'ilamb' not in attrs: attrs['ilamb'] = ''
        attrs['ilamb'] += "decycle(); "
        self.ds[self.varname].attrs = attrs
        return self

    def cycle(self):
        """Compute the annual cycle.

        """
        varname = self.varname
        ds = self.ds.groupby('time.month').mean(dim='time').rename({'month':'time'})
        ds['time'] = ds['time'].astype(float)
        tm = xr.DataArray(data=np.asarray([31,28,31,30,31,30,31,31,30,31,30,31],dtype='float'),
                          dims=['time'],coords={'time':ds['time']})
        cm = self.ds['cell_measure'] if 'cell_measure' in self.ds else None
        v = Variable(da=ds[varname],varname=varname,time_measure=tm,cell_measure=cm)
        attrs = self.ds[varname].attrs
        if 'ilamb' not in attrs: attrs['ilamb'] = ''
        attrs['ilamb'] += "cycle(); "
        if 'bounds' in attrs: v.ds[attrs['bounds']] = ds[attrs['bounds']]
        v.ds[varname].attrs = attrs
        return v

    def maxMonth(self):
        """

        """
        assert 'time' in self.ds
        v = self if self.ds['time'].size == 12 else self.cycle()
        da = v.ds[self.varname]
        da = xr.where(da.isnull().all(dim='time'),np.nan,xr.where(da.isnull(),0,da).argmax(dim='time'))
        cm = self.ds['cell_measure'] if 'cell_measure' in self.ds else None
        v = Variable(da = da, varname = "maxmonth", cell_measure = cm)
        v.ds[v.varname].attrs['units'] = 'month'
        return v
    
    def correlation(self,v,dim,region=None):
        """Compute the correlation is the specified dimension.

        """
        self,v = align_latlon(self,v)
        dims = []
        if dim ==  'time': dims = ['time']
        if dim == 'space': dims = [self.lat_name,self.lon_name]
        sds = self.ds if region is None else ilamb_regions.getMask(region,self)    
        vds =    v.ds if region is None else ilamb_regions.getMask(region,   v)    
        r = xr.corr(sds[self.varname],vds[v.varname],dim=dims)
        dims = ["'%s'" % d for d in dims]
        attrs = {}
        attrs['ilamb'] = "correlation(%s,%s,dim=[%s]); " % (self.varname,v.varname,",".join(dims))
        r.attrs = attrs
        tm = self.ds['time_measure'] if ('time_measure' in self.ds and 'time' in r.dims) else None
        cm = self.ds['cell_measure'] if ('cell_measure' in self.ds and (self.lat_name in r.dims and
                                                                        self.lon_name in r.dims)) else None
        r = Variable(da = r, varname = "corr_%s_%s" % (self.varname,v.varname),
                     cell_measure = cm, time_measure = tm)
        if r.ds[r.varname].size == 1: r = float(r.ds[r.varname])
        return r

    def nestSpatialGrids(self,*args):
        """
        """
        if self.sites():
            args = [self,]+list(args)
            for a in args:
                assert a.sites()
            return args
        lat = self.ds[self.lat_name].values
        lon = self.ds[self.lon_name].values
        for v in args:
            lat = np.union1d(lat,v.ds[v.lat_name])
            lon = np.union1d(lon,v.ds[v.lon_name])
        out = [self.interpolate(lat=lat,lon=lon)]
        for v in args: out.append(v.interpolate(lat=lat,lon=lon))
        return out

    def interpolate(self,lat=None,lon=None,res=None,**kwargs):
        """
        """
        varname = self.varname
        method = kwargs.get("method","nearest")
        if res is not None:
            lat = self.ds[self.lat_name]
            lat = np.linspace(lat.min(),lat.max(),int((lat.max()-lat.min())/res)+1)
            lon = self.ds[self.lon_name]
            lon = np.linspace(lon.min(),lon.max(),int((lon.max()-lon.min())/res)+1)
        assert lat is not None
        assert lon is not None
        dx = Variable(da = self.ds['cell_measure'],varname = "cell_fraction")
        self.ds['cell_fraction'] = (dx.ds['cell_fraction']/dx.ds['cell_measure']).clip(0,1)
        ds = self.ds.interp(coords={self.lat_name:lat,self.lon_name:lon},method=method)        
        tm = self.ds['time_measure'] if ('time_measure' in self.ds and 'time' in self.ds[varname].dims) else None
        out = Variable(da = ds[varname], varname = varname, time_measure = tm)
        out.ds['cell_measure'] *= ds['cell_fraction']        
        out.ds[varname].attrs = dict(self.ds[varname].attrs)
        for skip in ["cell_measures","cell_methods"]:
            if skip in out.ds[varname].attrs: out.ds[varname].attrs.pop(skip)
        if 'ilamb' not in out.ds[varname].attrs: out.ds[varname].attrs['ilamb'] = ''
        out.ds[varname].attrs['ilamb'] += "interpolate(); "
        if "bounds" in self.ds[varname].attrs and self.ds[varname].attrs['bounds'] in ds:
            bnd_name = varname+"_bnds"
            out.ds[bnd_name] = ds[self.ds[varname].attrs['bounds']]
            out.ds[varname].attrs['bounds'] = bnd_name
        return out
   
    def _createTimeMeasure(self):
        """Create the time measures from the bounds if present.
        """
        da = self.ds[self.varname]
        if "time" not in da.dims:
            msg = "To _createTimeMeasure you must have a coordinate named 'time'"
            raise ValueError(msg)
        tb_name = None
        if "bounds" in da['time'].attrs: tb_name = da.time.attrs['bounds']
        if tb_name is not None and tb_name in self.ds:
            dt = self.ds[tb_name]
            nb = dt.dims[-1]
            dt = dt.diff(nb).squeeze()
            dt *= 1e-9/86400 # [ns] to [d]
            self.ds['time_measure'] = dt.astype('float')
        else:
            dt = da['time'].diff(dim='time').mean()
            self.ds['time_measure'] = da.time.copy(data=[dt]*da.time.size)
        if 'ilamb' not in da.attrs: self.ds[self.varname].attrs['ilamb'] = ''
        self.ds[self.varname].attrs['ilamb'] += "_createTimeMeasure(); "
    
    def _createCellMeasure(self):
        """Creates the cell measures from the bounds if present.

        If no cell measures were passed into the constructor, then we
        will construct them when/if needed. First we look for 'bounds'
        in the spatial dimensions and use those to compute areas. If
        none are found, then we approximate cell sizes by differences
        with neighboring cells.
        """
        ms = None
        earth_rad = 6.371e6 # [m]
        latb_name = lonb_name = None
        da = self.ds[self.varname]
        if "bounds" in self.ds[self.lat_name].attrs: latb_name = da[self.lat_name].attrs['bounds']
        if "bounds" in self.ds[self.lon_name].attrs: lonb_name = da[self.lon_name].attrs['bounds']
        if (latb_name is not None and latb_name in self.ds and
            lonb_name is not None and lonb_name in self.ds and
            self.ds is not None):
            xb =        self.ds[lonb_name]*np.pi/180
            yb = np.sin(self.ds[latb_name]*np.pi/180)
            nb = xb.dims[-1]
            dx = earth_rad * xb.diff(nb).squeeze()
            dy = earth_rad * yb.diff(nb).squeeze()
            ms = dy*dx
        if ms is None:
            if self.is_regular:
                xb = da[self.lon_name].values
                yb = da[self.lat_name].values
                dx = 0.5*(xb[:-1]+xb[1:])
                dy = 0.5*(yb[:-1]+yb[1:])
                xb = np.vstack([np.hstack([xb[0]-0.5*(xb[1]-xb[0]),dx]),
                                np.hstack([dx,xb[-1]+0.5*(xb[-1]-xb[-2])])]).T
                yb = np.vstack([np.hstack([yb[0]-0.5*(yb[1]-yb[0]),dy]),
                                np.hstack([dy,yb[-1]+0.5*(yb[-1]-yb[-2])])]).T
                xb = xb.clip(0,360) if xb.max() > 180 else xb.clip(-180,180)
                yb = yb.clip(-90,90)
                xb =        xb*np.pi/180
                yb = np.sin(yb*np.pi/180)
                dx = earth_rad*np.diff(xb,axis=1).squeeze()
                dy = earth_rad*np.diff(yb,axis=1).squeeze()
                dx = xr.DataArray(data=np.abs(dx),dims=[self.lon_name],coords={self.lon_name:da[self.lon_name]})
                dy = xr.DataArray(data=np.abs(dy),dims=[self.lat_name],coords={self.lat_name:da[self.lat_name]})
                ms = dy*dx
            else:
                msg = "_createCellMeasure() for irregular grids is not implemented"
                raise ValueError(msg)
        
        if 'ilamb' not in da.attrs: self.ds[self.varname].attrs['ilamb'] = ''
        self.ds[self.varname].attrs['ilamb'] += "_createCellMeasure(); "
        self.ds['cell_measure'] = ms

if __name__ == "__main__":

    def test_timeint():
        # reference dataset with no cell measures
        fn = "/home/nate/data/ILAMB/DATA/gpp/FLUXCOM/tmp.nc"
        v  = Variable(filename = fn, varname = "gpp")
        vt = v.integrate(dim='time',mean=True).convert("g m-2 d-1")
        vt.plot(cmap="Greens",vmin=0); plt.show()
        # the model needs them and is not masked where the measures
        # are 0, like over oceans
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01")
        vt = v.integrate(dim='time',mean=True).convert("g m-2 d-1")
        vt.plot(cmap="Greens",vmin=0); plt.show()
        # in order to get the proper masking, we need to associate
        # cell measures
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        dx = Variable(filename = fn,varname = "areacella").convert("m2")
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        df = Variable(filename = fn,varname = "sftlf").convert("1")    
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01",
                      cell_measure = dx.ds['areacella'] * df.ds['sftlf'])
        vt = v.integrate(dim='time',mean=True).convert("g m-2 d-1")
        vt.plot(cmap="Greens",vmin=0); plt.show()
        
    def test_spaceint():
        fig,ax = plt.subplots(tight_layout=True)
        # reference dataset with no cell measures
        fn = "/home/nate/data/ILAMB/DATA/gpp/FLUXCOM/tmp.nc"
        v  = Variable(filename = fn, varname = "gpp").convert('g m-2 d-1')
        vs = v.integrate(dim='space',mean=True)
        vs.plot(ax=ax,label='reference')
        # the model needs measures and is not masked where the
        # measures are 0, like over oceans, unlike netCDF4 python
        # bindings, they are not based on numpy masked arrays and so
        # we have to create our own method of ignoring areas.
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01").convert('g m-2 d-1')
        vs = v.integrate(dim='space',mean=True)
        vs.plot(ax=ax,label='missing model measures')
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        dx = Variable(filename = fn,varname = "areacella").convert("m2")
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        df = Variable(filename = fn,varname = "sftlf").convert("1")    
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01",
                      cell_measure = dx.ds['areacella'] * df.ds['sftlf'])
        vs = v.integrate(dim='space',mean=True).convert('g m-2 d-1')
        vs.plot(ax=ax,label='with correct model measures')
        plt.legend()
        plt.show() # requires nc-time-axis
        plt.close()

    def test_detrend():
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        dx = Variable(filename = fn,varname = "areacella").convert("m2")
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        df = Variable(filename = fn,varname = "sftlf").convert("1")    
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01",
                      cell_measure = dx.ds['areacella'] * df.ds['sftlf'])
        vs = v.integrate(dim='space',mean=True).convert('g m-2 d-1')
        fig,ax = plt.subplots(tight_layout=True)
        vs.plot(ax=ax,label='original')
        vs.detrend(dim=['time'])
        print(vs.ds.gpp_sint.ilamb)
        vs.plot(ax=ax,label='detrended')
        v.detrend(dim='time')
        vs = v.integrate(dim='space',mean=True).convert('g m-2 d-1')
        vs.plot(ax=ax,label='detrended multi-dim',linestyle='--')
        print(vs.ds.gpp_sint.ilamb)
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01",
                      cell_measure = dx.ds['areacella'] * df.ds['sftlf'])
        vs = v.integrate(dim='space',mean=True).convert('g m-2 d-1')
        vs.detrend(dim=['time'],degree=3)
        print(vs.ds.gpp_sint.ilamb)
        vs.plot(ax=ax,label='detrended cubic',linestyle=':')
        plt.legend()
        plt.show()

    def test_decycle():
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        dx = Variable(filename = fn,varname = "areacella").convert("m2")
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        df = Variable(filename = fn,varname = "sftlf").convert("1")    
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01",
                      cell_measure = dx.ds['areacella'] * df.ds['sftlf'])
        vs = v.integrate(dim='space',mean=True).convert('g m-2 d-1')
        fig,ax = plt.subplots(tight_layout=True)
        vs.plot(ax=ax,label='original')
        vs.detrend(dim='time')
        print(vs.ds.gpp_sint.ilamb)
        vs.plot(ax=ax,label='detrend')
        vs.decycle()
        print(vs.ds.gpp_sint.ilamb)
        vs.plot(ax=ax,label='decycle')
        vs = v.integrate(dim='space',mean=True).convert('g m-2 d-1').detrend(dim='time').decycle()
        print(vs.ds.gpp_sint.ilamb)
        vs.plot(ax=ax,label='composed',linestyle=':')
        plt.legend()
        plt.show()

    def test_plot():
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        dx = Variable(filename = fn,varname = "areacella").convert("m2")
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        df = Variable(filename = fn,varname = "sftlf").convert("1")    
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01",
                      cell_measure = dx.ds['areacella'] * df.ds['sftlf'])
        vt = v.integrate(dim='time',mean=True).convert('g m-2 d-1')
        vt.plot(cmap="Greens",vmin=0,region="nhsa"); plt.show()
        vt.plot(cmap="Greens",vmin=0,region="shsa"); plt.show()
        vt.plot(cmap="Greens",vmin=0); plt.show()

    def test_cellmeasures():
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        dx = Variable(filename = fn,varname = "areacella").convert("m2")
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        df = Variable(filename = fn,varname = "sftlf").convert("1")    
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01")
        e  = np.abs((v.ds.cell_measure - dx.ds.areacella)/dx.ds.areacella)
        assert e.max()<1e-4
        v.ds = v.ds.drop(["cell_measure","lat_bnds","lon_bnds"])
        v._createCellMeasure()
        e  = np.abs((v.ds.cell_measure - dx.ds.areacella)/dx.ds.areacella)
        assert e.mean()<1e-2 # less strict as this is more approximate
        
    def test_interp():
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        dx = Variable(filename = fn,varname = "areacella").convert("m2")
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc"
        df = Variable(filename = fn,varname = "sftlf").convert("1")    
        fn = "/home/nate/data/ILAMB/MODELS/CMIP6/CanESM5/gpp_Lmon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc"
        v  = Variable(filename = fn, varname = "gpp", t0 = "1990-01-01", tf = "2000-01-01",
                      cell_measure = dx.ds['areacella'] * df.ds['sftlf'])
        vt = v.integrate(dim='time',mean=True).convert('g m-2 d-1')
        lat = np.linspace(-90, 90,361); lat = 0.5*(lat[:-1]+lat[1:])
        lon = np.linspace(  0,360,721); lon = 0.5*(lon[:-1]+lon[1:])
        Vt = vt.interpolate(lat=lat,lon=lon)
        vt.plot(cmap="Greens",vmin=0); plt.show()
        Vt.plot(cmap="Greens",vmin=0); plt.show()
        vt = vt.integrate(dim='space').convert("Pg yr-1").ds.gpp_tint_sint.values
        Vt = Vt.integrate(dim='space').convert("Pg yr-1").ds.gpp_tint_sint.values
        e  = np.abs((Vt-vt)/vt)
        assert e < 1e-2

    test_timeint()
    test_spaceint()
    test_detrend()
    test_decycle()
    test_plot()
    test_interp()
    test_cellmeasures()

    
        
