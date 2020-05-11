from cf_units import Unit
import xarray as xr
import numpy as np

@xr.register_dataarray_accessor('ilamb')
class ilamb_variable:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.measure = None
        self.bounds = None
        
    def convert(self,unit,density=998.2):
        """Convert the variable to a given unit.

        We use the UDUNITS2 library via the cf_units python interface
        to convert the unit. Additional support is provided for unit
        conversions in which substance information is required, such
        as in precipitation where it is common to have data in a mass
        rate per unit area [kg s-1 m-2] and desire it in a linear rate
        [m s-1]. Conversion occurs in place, but also returns the
        DataArray object so that the user can chain calls together.

        Parameters
        ----------
        unit : str
            the desired unit
        density : float, optional
            the mass density in [kg m-3] to use when converting,
            defaults to that of water

        Returns
        -------
        self : xarray.DataArray
        """
        if 'units' not in self._obj.attrs:
            msg = "Cannot convert the units of a DataArray lacking the 'units' attribute"
            raise ValueError(msg)
        src_unit = Unit(self._obj.units)
        tar_unit = Unit(unit)
        
        # Define some generic quantities
        linear            = Unit("m")
        linear_rate       = Unit("m s-1")
        area_density      = Unit("kg m-2")
        area_density_rate = Unit("kg m-2 s-1")
        mass_density      = Unit("kg m-3")

        # Do we need to multiply by density?
        if ( (src_unit.is_convertible(linear_rate) and tar_unit.is_convertible(area_density_rate)) or
             (src_unit.is_convertible(linear     ) and tar_unit.is_convertible(area_density     )) ):
            with np.errstate(over='ignore',under='ignore'):
                self._obj.data *= density
            src_unit *= mass_density

        # Do we need to divide by density?
        if ( (tar_unit.is_convertible(linear_rate) and src_unit.is_convertible(area_density_rate)) or
             (tar_unit.is_convertible(linear     ) and src_unit.is_convertible(area_density     )) ):
            with np.errstate(over='ignore',under='ignore'):
                self._obj.data /= density
            src_unit /= mass_density

        # Convert the unit
        with np.errstate(over='ignore',under='ignore'):
            src_unit.convert(self._obj.data,tar_unit,inplace=True)
        self._obj.attrs['units'] = unit
        return self._obj

    def add_measure(self,area_filename,fraction_filename=None):

        # check that the measure is in the attributes
        if 'cell_measures' not in self._obj.attrs:
            msg = "DataArray does not contain the 'cell_measures' attribute"
            raise ValueError(msg)

        # try to get the cell areas from the file
        measure_name = self._obj.cell_measures.split(":")[1].strip()
        with xr.open_dataset(area_filename) as ds:
            if measure_name not in ds:
                msg = "The cell_measures: %s is not found in %s" % (measure_name,area_filename)
                raise ValueError(msg)
            self.measure = ds[measure_name]
            
        # optionally multiply by cell fractions
        if fraction_filename is None: return
        fraction_name = "sftlf"
        with xr.open_dataset(fraction_filename) as ds:
            if fraction_name not in ds:
                msg = "The fraction variable sftlf is not found in %s" % (area_filename)
                raise ValueError(msg)
            self.measure *= ds[fraction_name].ilamb.convert("1")

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
    nbp.ilamb.add_bounds(ds)       # look in the dataset for 'bounds' on the time array
    nbp.ilamb.add_measure(areacella_file,sftlf_file)  # give filenames in which we look for measures
    mean = nbp.ilamb.integrate_space()  
    mass = mean.ilamb.cumsum()
    mass.ilamb.convert("Pg")
    mass.plot()
    plt.show()

    """

    Do you know if there is a way to get access to the parent dataset object in a dataarray?
    You may be able to write it in a way that acts on the Dataset rather than the DataArray...?
    

    value = mean.ilamb.integrate_time(initial_time='1990-01-01',
                                      final_time='1995-01-01',
                                      mean=True)
    value.ilamb.convert("Pg yr-1")
    """

