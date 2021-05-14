from cf_units import Unit
import xarray as xr
import numpy as np

"""

"""
def _stripConstants(unit):
    """Sometimes cf_units gives us units with strange constants in front,
    remove any token purely numeric.
    """
    T = str(unit).split()
    out = []
    for t in T:
        try:
            junk = float(t)
        except:
            out.append(t)
    return " ".join(out)

@xr.register_dataarray_accessor('ilamb')
class ilamb_variable:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.measure = None
        self.fraction = None
        self.bounds = {}

    def timeScale(self):
        """Returns mean time scale of the data.
        """
        dt = self._obj.time.diff('time').mean()
        return dt.data.astype(float)*1e-9/86400
        
    def convert(self,unit,density=998.2,molar_mass=12.011):
        """
        - handles ( M L-2 T-1 ) --> ( L T-1 )
        - handles (       mol ) --> (     M )
        """
        if 'units' not in self._obj.attrs:
            msg = "Cannot convert the units of a DataArray lacking the 'units' attribute"
            raise ValueError(msg)
        src_unit = Unit(self._obj.units)
        tar_unit = Unit(unit)        
        mass_density = Unit("kg m-3")
        molar_density = Unit("g mol-1")
        if ((src_unit/tar_unit)/mass_density).is_dimensionless():
            self._obj.data /= density
            src_unit /= mass_density
        elif ((tar_unit/src_unit)/mass_density).is_dimensionless():
            self._obj.data *= density
            src_unit *= mass_density
        if ((src_unit/tar_unit)/molar_density).is_dimensionless():
            self._obj.data /= molar_mass
            src_unit /= molar_density
        elif ((tar_unit/src_unit)/molar_density).is_dimensionless():
            self._obj.data *= molar_mass
            src_unit *= molar_density
        src_unit.convert(self._obj.data,tar_unit,inplace=True)
        self._obj.attrs['units'] = unit
        return self._obj

    def setMeasure(self,area_filename=None,fraction_filename=None,area=None,fraction=None):
        """
        - should check sizes
        """
        if area is not None:
            self.measure = area.ilamb.convert("m2")
        elif area_filename is not None:
            with xr.open_dataset(area_filename) as ds:
                measure_name = [v for v in ["areacella","area","areacello"] if v in ds]
                if len(measure_name)==0:
                    msg = "Cannot find ['areacella','area','areacello'] in %s" % area_filename
                    raise ValueError(msg)
                self.measure = ds[measure_name[0]].ilamb.convert("m2")
        if fraction is not None:
            self.fraction = fraction.ilamb.convert("1")
            self.measure = self.measure*self.fraction
        elif fraction_filename is not None:
            with xr.open_dataset(fraction_filename) as ds:
                frac_name = [v for v in ["sftlf","landfrac","sftof"] if v in ds]
                if len(frac_name)==0:
                    msg = "Cannot find ['sftlf','landfrac','sftof'] in %s" % fraction_filename
                    raise ValueError(msg)
                self.fraction = ds[frac_name[0]].ilamb.convert("1")
                self.measure = self.measure*self.fraction
        
    def setBounds(self,dset):
        """
        - should check sizes and that values fall inside bounds
        """
        for d in self._obj.dims:
            if "bounds" in self._obj[d].attrs:
                bnd = self._obj[d].attrs["bounds"]
                if bnd in dset: self.bounds[d] = dset[bnd]
        
    def integrateInSpace(self,mean=False):
        """
        - need to include regions, and perhaps lat/lon bounds
        """
        if self.measure is None:
            msg = "To integrateInSpace you must first add cell areas with setMeasure()"
            raise ValueError(msg)

        V,A = xr.align(self._obj,self.measure,join='override',copy=False)
        out = (V*A).sum(A.dims)
        unit = Unit(self._obj.units)
        out.attrs['units'] = str(unit)
        if mean:
            out /= self.measure.sum()
        else:
            unit *= Unit("m2")
            out.attrs['units'] = str(unit)
            out.ilamb.convert(_stripConstants(unit))
        for d in out.dims: out.ilamb.bounds[d] = self.bounds[d]
        return out

    def integrateInTime(self,mean=False):
        """
        - need to add initial/final time
        """
        if "time" not in self.bounds:
            msg = "To integrateInTime you must first add bounds on the time intervals with setBounds()"
            raise ValueError(msg)
        dt = self.bounds['time'][:,1]-self.bounds['time'][:,0]
        dt.data = dt.data.astype(float)*1e-9/86400
        out = (self._obj*dt).sum(dt.dims)
        unit = Unit(self._obj.units)
        out.attrs['units'] = str(unit)
        if mean:
            out /= dt.sum()
        else:
            unit *= Unit("d")
            out.attrs['units'] = str(unit)
            out.ilamb.convert(_stripConstants(unit))
        for d in out.dims: out.ilamb.bounds[d] = self.bounds[d]
        return out
        
    def accumulateInTime(self):
        """
        """
        if "time" not in self.bounds:
            msg = "To accumulateInTime you must first add bounds on the time intervals with setBounds()"
            raise ValueError(msg)
        dt = self.bounds['time'][:,1]-self.bounds['time'][:,0]
        dt.data = dt.data.astype(float)*1e-9/86400
        out = (self._obj*dt).cumsum(dt.dims)
        unit = Unit(self._obj.units)*Unit("d")
        out.attrs['units'] = str(unit)
        out.ilamb.convert(_stripConstants(unit))
        for d in out.dims: out.ilamb.bounds[d] = self.bounds[d]
        return out                

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
    nbp.ilamb.setMeasure(areacella_file,sftlf_file)  # give filenames in which we look for measures
    nbp.ilamb.setBounds(ds) # harvest the bounds information from the dataset
    mean = nbp.ilamb.integrateInSpace()
    acc = mean.ilamb.accumulateInTime()
    acc.ilamb.convert("Pg")
    ann = acc.coarsen(time=12,boundary="trim").mean()
    ann.ilamb.convert("Pg")
    acc.plot()
    ann.plot()
    plt.show()
