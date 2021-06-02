from cf_units import Unit
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.tri import Triangulation
from scipy.interpolate import NearestNDInterpolator
from Regions import Regions

class Variable():

    def __init__(self,**kwargs):

        self.filename = kwargs.get("filename",None)
        self.varname = kwargs.get("varname",None)
        self.ds = kwargs.get("ds",None)
        self.da = kwargs.get("da",None)
        self.dx = kwargs.get("dx",None)  # the area of the cells [m2]
        self.dt = kwargs.get("dt",None)  # the length of the time intervals [d]
        t0 = kwargs.get("t0",None)
        tf = kwargs.get("tf",None)

        # These methods do not read memory, the dataset is left
        # unchanged in case we need something out of it, the dataarray
        # is what we will operate on moving forward.
        if self.filename is not None:
            self.ds = xr.open_dataset(self.filename)
            self.da = self.ds[self.varname]

            # To prevent more memory being read than intended, trim away
            # times we don't need if specified
            if 'time' in self.da.dims:
                self.da = self.da.sel(time=slice(t0,tf))

        # Consistency check on times
        if self.dt is not None and (t0 is not None or tf is not None): self.dt = self.dt.sel(time=slice(t0,tf))
        assert self.da is not None

        # Is there a 'lat' or 'lon' dimension? If so, get the names
        self.lat_name = [d for d in self.da.dims if ('lat' in d.lower() or 'j' == d or 'y' == d)]
        self.lon_name = [d for d in self.da.dims if ('lon' in d.lower() or 'i' == d or 'x' == d)]
        if len(self.lat_name) > 1 or len(self.lon_name) > 1:
            msg  = "Ambiguity in which data array dimensions are spatial coordinates: "
            msg += "lat = ['%s'], lon = ['%s']" % (",".join(self.lat_name),",".join(self.lon_name))
            raise ValueError(msg)
        
        # If the data types of these 'lat' and 'lon' dimensions are
        # integers, they are index sets and therefore the grid is
        # irregular, for now just flag
        self.lat_name = self.lat_name[0] if self.lat_name else None
        self.lon_name = self.lon_name[0] if self.lon_name else None
        self.is_regular = True
        if self.lat_name and self.lon_name:
            if (np.issubdtype(self.da[self.lat_name].dtype,np.integer) and
                np.issubdtype(self.da[self.lon_name].dtype,np.integer)):
                self.is_regular = False
        
    def __str__(self):
        out  = self.da.__str__()
        out += "\nStatistics:"
        out += "\n    {0:<16}{1:.6e}".format("minimum:",self.da.min().data)
        out += "\n    {0:<16}{1:.6e}".format("mean:",self.da.mean().data)
        out += "\n    {0:<16}{1:.6e}".format("maximum:",self.da.max().data)
        return out

    def convert(self,unit,density=998.2,molar_mass=12.011):
        """Using cf_units (UDUNITS2) convert the unit in place
        - handles ( M L-2 T-1 ) --> ( L T-1 ), assuming water
        - handles (       mol ) --> (     M ), assuming carbon
        """
        def _stripConstants(unit):
            """Sometimes cf_units gives us units with strange constants in front,
            remove any token which is purely numeric.
            """
            T = str(unit).split()
            out = []
            for t in T:
                try:
                    junk = float(t)
                except:
                    out.append(t)
            return " ".join(out)
        if 'units' not in self.da.attrs:
            msg = "Cannot convert the units of the DataArray lacking the 'units' attribute"
            raise ValueError(msg)
        src_unit = Unit(self.da.units)
        tar_unit = Unit(unit)
        mass_density = Unit("kg m-3")
        molar_density = Unit("g mol-1")
        if ((src_unit/tar_unit)/mass_density).is_dimensionless():
            self.da.data /= density
            src_unit /= mass_density
        elif ((tar_unit/src_unit)/mass_density).is_dimensionless():
            self.da.data *= density
            src_unit *= mass_density
        if ((src_unit/tar_unit)/molar_density).is_dimensionless():
            self.da.data /= molar_mass
            src_unit /= molar_density
        elif ((tar_unit/src_unit)/molar_density).is_dimensionless():
            self.da.data *= molar_mass
            src_unit *= molar_density
        src_unit.convert(self.da.data,tar_unit,inplace=True)
        self.da.attrs['units'] = unit
        return self

    def _createTimeBounds(self):
        """
        """
        if "time" not in self.da.coords:
            msg = "To _createTimeBounds you must have a coordinate named 'time'"
            raise ValueError(msg)
        tb_name = None
        if "bounds" in self.da['time'].attrs: tb_name = self.da.time.attrs['bounds']
        if tb_name is not None and self.ds is not None and tb_name in self.ds:
            dt = self.ds[tb_name]
            nb = dt.dims[-1]
            dt = dt.diff(nb).squeeze()
            dt *= 1e-9/86400 # [ns] to [d]
            self.dt = dt.astype('float')
        else:
            msg = "Not implemented"
            raise ValueError(msg)

    def _createMeasure(self):
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
        if "bounds" in self.da[self.lat_name].attrs: latb_name = self.da[self.lat_name].attrs['bounds']
        if "bounds" in self.da[self.lon_name].attrs: lonb_name = self.da[self.lon_name].attrs['bounds']
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
                xb = self.da[self.lon_name].values
                yb = self.da[self.lat_name].values
                xb = np.vstack([np.hstack([xb[0]-0.5*(xb[1]-xb[0]),xb[:-1]]),
                                np.hstack([xb[1:],xb[-1]+0.5*(xb[-1]-xb[-2])])]).T
                yb = np.vstack([np.hstack([yb[0]-0.5*(yb[1]-yb[0]),yb[:-1]]),
                                np.hstack([yb[1:],yb[-1]+0.5*(yb[-1]-yb[-2])])]).T
                xb = xb.clip(0,360) if xb.max() >= 180 else xb.clip(-180,180)
                yb = yb.clip(-90,90)
                xb =        xb*np.pi/180
                yb = np.sin(yb*np.pi/180)
                dx = earth_rad*np.diff(xb,axis=1).squeeze()
                dy = earth_rad*np.diff(yb,axis=1).squeeze()
                dx = xr.DataArray(data=dx,dims={self.lon_name:self.da[self.lon_name]})
                dy = xr.DataArray(data=dy,dims={self.lat_name:self.da[self.lat_name]})
                ms = dy*dx
            else:
                msg = "Not implemented"
                raise ValueError(msg)
            
        if 'ilamb' not in self.da.attrs: self.da.attrs['ilamb'] = ''
        self.da.attrs['ilamb'] += "_createMeasure(); "
        self.dx = ms
        
    def integrateInTime(self,t0=None,tf=None,mean=False):
        """
        """
        if "time" not in self.da.coords:
            msg = "To integrateInTime you must have a coordinate named 'time'"
            raise ValueError(msg)
        if self.dt is None: self._createTimeBounds()
        assert self.dt is not None
        da = self.da.sel(time=slice(t0,tf))
        dt = self.dt.sel(time=slice(t0,tf))
        out = (da*dt).sum(dt.dims,min_count=1)
        units = Unit(self.da.units)
        out.attrs = {key:a for key,a in da.attrs.items() if "time" not in key}
        if 'ilamb' not in out.attrs: out.attrs['ilamb'] = ''
        out.attrs['ilamb'] += "integrateInTime(t0='%s',tf='%s',mean=%s); " % (t0,tf,mean)
        v = Variable(ds=self.ds,da=out,dx=self.dx)
        if mean:
            out /= dt.sel(time=slice(da.time[0],da.time[-1])).sum()
        else:
            units *= Unit("d")
            out.attrs['units'] = str(units)
            v.convert(_stripConstants(units))
        return v

    def integrateInSpace(self,region=None,mean=False):
        """
        """
        if self.dx is None: self._createMeasure()
        assert self.dx is not None            
        v,dx = xr.align(self.da,xr.where(self.dx<1,np.nan,self.dx),join='override',copy=False)
        out = (v*dx).sum(dx.dims)
        units = Unit(self.da.units)
        out.attrs = {key:a for key,a in v.attrs.items() if "cell_" not in key}
        if 'ilamb' not in out.attrs: out.attrs['ilamb'] = ''
        out.attrs['ilamb'] += "integrateInSpace(mean=%s); " % (mean)
        v = Variable(ds=self.ds,da=out,dt=self.dt)
        if mean:
            mask = self.da.isnull()
            dims = set(mask.dims).difference(set(dx.dims))
            if dims: mask = mask.all(dims)
            out /= (dx*(mask==False)).sum()
        else:
            units *= Unit('m2')
            out.attrs['units'] = str(units)
            v.convert(_stripConstants(units))
        return v

    def _interpolateIrregular(self,res=None):
        """the 'longitude', 'latitiude' things need detected

        If no resolution is given, we will guess it by (1) creating a
        triangulation of the latitudes and longitudes (2) finding the
        mean triangle area and (3) picking a resolution based on the
        square root of twice this area.

        """
        ds = self.ds
        da = self.da
        space = [self.lat_name,self.lon_name]
        lon0 = ds['longitude'].to_masked_array().data.flatten()
        lat0 = ds['latitude' ].to_masked_array().data.flatten()
        shp0 = [da[d].size for d in da.dims if d not in space]
        shp0.append(np.prod([da[d].size for d in da.dims if d in space]))
        data0 = da.to_masked_array().reshape(shp0)
        if res is None:
            T = Triangulation(lon0,lat0)
            A  = np.cross(np.asarray([T.x[T.triangles[:,1]]-T.x[T.triangles[:,0]],
                                      T.y[T.triangles[:,1]]-T.y[T.triangles[:,0]]]).T,
                          np.asarray([T.x[T.triangles[:,2]]-T.x[T.triangles[:,0]],
                                      T.y[T.triangles[:,2]]-T.y[T.triangles[:,0]]]).T).mean()
            res = np.sqrt(2*A)
        lat = np.linspace(lat0.min(),lat0.max(),int((lat0.max()-lat0.min())/res))
        lon = np.linspace(lon0.min(),lon0.max(),int((lon0.max()-lon0.min())/res))
        shp = [da[d].size for d in da.dims if d not in space] + [lat.size,lon.size]
        lat2d,lon2d = np.meshgrid(lat,lon,indexing='ij')
        interp = NearestNDInterpolator(np.asarray([lon0,lat0]).T,data0.T)
        data = interp(lon2d.flatten(),lat2d.flatten()).T.reshape(shp)
        dims = [d for d in da.dims if d not in space] + ['lat','lon']
        coords = {c:da[c] for c in da.dims if c not in space}
        coords['lat'] = lat
        coords['lon'] = lon
        da = xr.DataArray(data=data,dims=dims,coords=coords)
        da.attrs = dict(self.da.attrs)
        if 'ilamb' not in da.attrs: da.attrs['ilamb'] = ''
        da.attrs['ilamb'] += "_interpolateIrregular(res=%.4g); " % (res)
        return Variable(ds=ds,da=da,varname=self.varname)

    def interpolate(self,lat=None,lon=None,res=None,**kwargs):
        """
        """
        if not self.is_regular: return self._interpolateIrregular(res)
        if res is not None:
            lat = self.da[self.lat_name]
            lat = np.linspace(lat.min(),lat.max(),int((lat.max()-lat.min())/res)+1)
            lon = self.da[self.lon_name]
            lon = np.linspace(lon.min(),lon.max(),int((lon.max()-lon.min())/res)+1)
        assert lat is not None
        assert lon is not None
        out = Variable(ds=self.ds,da=self.da.interp(coords={self.lat_name:lat,self.lon_name:lon}))
        out.da.attrs = dict(self.da.attrs)
        for skip in ["cell_measures","cell_methods"]:
            if skip in out.da.attrs: out.da.attrs.pop(skip)
        if 'ilamb' not in out.da.attrs: out.da.attrs['ilamb'] = ''
        out.da.attrs['ilamb'] += "_interpolate(); "
        return out
    
    def plot(self,**kwargs):
        """
        """
        if self.da.ndim == 1:
            self.da.plot(**kwargs)
        elif self.da.ndim == 2:
            fig,ax = plt.subplots(dpi=200,subplot_kw={'projection':ccrs.Robinson()})
            da = self.da
            if self.dx is not None: da = xr.where(self.dx<1,np.nan,self.da)
            p = da.plot(ax=ax,transform=ccrs.PlateCarree(),**kwargs)
            ax.add_feature(cfeature.NaturalEarthFeature('physical','land','110m',
                                                        edgecolor='face',
                                                        facecolor='0.875'),zorder=-1)
            ax.add_feature(cfeature.NaturalEarthFeature('physical','ocean','110m',
                                                        edgecolor='face',
                                                        facecolor='0.750'),zorder=-1)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    if 1:
        root = os.path.join(os.environ['ILAMB_ROOT'],'MODELS/CMIP6/CESM2')
        dx = Variable(filename = os.path.join(root,"areacella_fx_CESM2_historical_r1i1p1f1_gn.nc"),
                      varname = "areacella").convert("m2")
        lf = Variable(filename = os.path.join(root,"sftlf_fx_CESM2_historical_r1i1p1f1_gn.nc"),
                      varname = "sftlf").convert("1")
        v = Variable(filename = os.path.join(root,"gpp_Lmon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"),
                     varname = "gpp",
                     t0 = '1980-01-01',
                     dx = dx.da * lf.da).convert("g m-2 d-1")
        print(v.integrateInTime(mean=True).integrateInSpace(mean=True))
        v = Variable(filename = os.path.join(root,"gpp_Lmon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"),
                     varname = "gpp",
                     t0 = '1980-01-01',
                     dx = dx.da).convert("g m-2 d-1")
        print(v.integrateInTime(mean=True).integrateInSpace(mean=True))
        v = Variable(filename = os.path.join(root,"gpp_Lmon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"),
                     varname = "gpp",
                     t0 = '1980-01-01').convert("g m-2 d-1")
        print(v.integrateInTime(mean=True).integrateInSpace(mean=True))

    
    if 1:
        root = "./DATA/CanESM5"
        dx_atm = Variable(filename = os.path.join(root,"areacella_fx_CanESM5_1pctCO2_r1i1p1f1_gn.nc"),
                          varname = "areacella").convert("m2").da
        dx_lnd = Variable(filename = os.path.join(root,"sftlf_fx_CanESM5_1pctCO2_r1i1p1f1_gn.nc"),
                          varname = "sftlf").convert("1").da * dx_atm
        dx_ocn = Variable(filename = os.path.join(root,"areacello_Ofx_CanESM5_1pctCO2_r1i1p1f1_gn.nc"),
                          varname = "areacello").convert("m2").da
        tas = Variable(filename = os.path.join(root,"tas_Amon_CanESM5_1pctCO2_r1i1p1f1_gn_185001-200012.nc"),
                       varname = "tas",t0 = "1980-01-01",dx = dx_atm).convert("degC")
        fgco2 = Variable(filename = os.path.join(root,"fgco2_Omon_CanESM5_1pctCO2_r1i1p1f1_gn_185001-200012.nc"),
                         varname = "fgco2",t0 = "1980-01-01",dx = dx_ocn)
        nbp = Variable(filename = os.path.join(root,"nbp_Lmon_CanESM5_1pctCO2_r1i1p1f1_gn_185001-200012.nc"),
                       varname = "nbp",t0 = "1980-01-01",dx = dx_lnd).convert('g m-2 d-1')

        for v in [tas,fgco2,nbp]:
            print(v.varname)
            print(v.integrateInTime(mean=True).integrateInSpace(mean=True))
            if not v.is_regular:
                u = v.interpolate()
                print(u.integrateInTime(mean=True).integrateInSpace(mean=True))

