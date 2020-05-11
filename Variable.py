from cf_units import Unit
import xarray as xr
import numpy as np

"""

"""

@xr.register_dataarray_accessor('ilamb')
class ilamb_variable:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.measure = None
        self.fraction = None
        self.bounds = {}
        
    def convert(self,unit,density=998.2):
        """
        - handles ( M L-2 T-1 ) --> ( L T-1 )
        """
        if 'units' not in self._obj.attrs:
            msg = "Cannot convert the units of a DataArray lacking the 'units' attribute"
            raise ValueError(msg)
        src_unit = Unit(self._obj.units)
        tar_unit = Unit(unit)        
        mass_density = Unit("kg m-3")
        print("(%s) (%s) (%s)" % (src_unit,tar_unit,src_unit/tar_unit),((src_unit/tar_unit)/mass_density).is_dimensionless())        
        if ((src_unit/tar_unit)/mass_density).is_dimensionless():
            self._obj.data /= density
            src_unit /= mass_density
        elif ((tar_unit/src_unit)/mass_density).is_dimensionless():
            self._obj.data *= density
            src_unit *= mass_density
        src_unit.convert(self._obj.data,tar_unit,inplace=True)
        self._obj.attrs['units'] = unit
        return self._obj

    def addMeasure(self,area_filename,fraction_filename=None):
        with xr.open_dataset(area_filename) as ds:
            measure_name = [v for v in ["areacella","area","areacello"] if v in ds]
            if len(measure_name)==0:
                msg = "Cannot find ['areacella','area','areacello'] in %s" % area_filename
                raise ValueError(msg)
            self.measure = ds[measure_name[0]].ilamb.convert("m2")
            

    def integrate_space(self,mean=False):

        if self.measure is None:
            msg = "Must call ilamb.add_measure() before you can call ilamb.integrate_space()"
            raise ValueError(msg)

        # area weighted sum, divided by area if taking a mean
        out = (self._obj*self.measure).sum(self.measure.dims)
        if mean: out /= self.measure.sum()

        # handle unit shifts, measure assumed in m2 if not given
        unit  = Unit(self._obj.units)
        if not mean:
            unit *= Unit(self.measure.units if "units" in self.measure.attrs else "m2")
        out.attrs['units'] = str(unit).replace("."," ")
        out.ilamb.bounds = self.bounds
        
        return out

    def integrate_time(self,initial_time=None,final_time=None,mean=False):

        if self.bounds is None:
            msg = "Must call ilamb.add_bounds() before you can call ilamb.integrate_time()"
            raise ValueError(msg)

        # do we need to subset?
        if initial_time is not None or final_time is not None:
            data = self._obj.loc[initial_time:final_time]
            dt   = self.bounds.loc[initial_time:final_time]
            ## look into self.bounds.isel(time=slice(initial_time, final_time))
        else:
            data = self._obj
            dt = self.bounds
            
        # area weighted sum, divided by area if taking a mean
        out = (data*dt).sum('time')
        if mean: out /= dt.sum()

        # handle unit shifts, bounds assumed in d if not given
        unit  = Unit(self._obj.units)
        if not mean:
            unit *= Unit(self.bounds.units if "units" in self.bounds.attrs else "d")
        out.attrs['units'] = str(unit).replace("."," ")
        out.ilamb.measure = self.measure
        
        return out

    def cumsum(self):
        
        if self.bounds is None:
            msg = "Must call ilamb.add_bounds() before you can call ilamb.cumsum()"
            raise ValueError(msg)

        out = (self._obj*self.bounds).cumsum(dim='time')
        unit  = Unit(self._obj.units)
        unit *= Unit(self.bounds.units if "units" in self.bounds.attrs else "d")
        out.attrs['units'] = str(unit).replace("."," ")
        out.ilamb.measure = self.measure
        return out
        
    def add_bounds(self,dataset):
        if not ('time' in dataset and 'time' in self._obj.coords): return
        t0 = dataset['time']
        t  = self._obj['time']
        if 'bounds' not in t.attrs: return
        if t0.size != t.size: return
        #if not np.allclose(t0,t): return # should check this but it breaks
        if t.bounds not in dataset: return
        self.bounds = np.diff(dataset[t.bounds].values,axis=1)
        self.bounds = xr.DataArray([bnd.total_seconds()/(24*3600) for bnd in self.bounds[:,0]],
                                   dims = ('time'),
                                   coords = {'time':t})
        self.bounds.attrs['units'] = 'd'

                
        
if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt
    
    nbp_file = os.path.join(os.environ["ILAMB_ROOT"],
                            "MODELS/esmHistorical/CESM1-BGC/CESM1-BGC",
                            "nbp/nbp_Lmon_CESM1-BGC_esmHistorical_r1i1p1_185001-200512.nc")
    areacella_file = os.path.join(os.environ["ILAMB_ROOT"],
                                  "MODELS/esmHistorical/CESM1-BGC/CESM1-BGC",
                                  "areacella/areacella_fx_CESM1-BGC_esmHistorical_r0i0p0.nc")
    sftlf_file = os.path.join(os.environ["ILAMB_ROOT"],
                              "MODELS/esmHistorical/CESM1-BGC/CESM1-BGC",
                              "sftlf/sftlf_fx_CESM1-BGC_esmHistorical_r0i0p0.nc")
    
    ds = xr.open_dataset(nbp_file) # open the nbp
    nbp = ds.nbp                   # pointer to the dataarray
    nbp.ilamb.addMeasure(areacella_file,sftlf_file)  # give filenames in which we look for measures
    
    #mean = nbp.ilamb.integrate_space()
    #mass = mean.ilamb.cumsum()
    #mass.ilamb.convert("Pg")
    #mass.plot()
    #plt.show()
