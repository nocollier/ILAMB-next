import os
from Variable import Variable
import numpy as np
import matplotlib.pyplot as plt

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
        b.ds[b.lon_name] = (b.ds[b.lon_name]+180)%360-180
        b.ds = b.ds.sortby(b.lon_name)
    return a,b

def ScoreBias(r0,c0,r=None,c=None,regions=None):
    """
    r0,c0 - original versions
    r,c - grid aligned versions
    """
    rm0 = r0.integrateInTime(mean=True)
    cm0 = c0.integrateInTime(mean=True)
    norm0 = r0.stddev() if r0.ds['time'].size > 1 else rm0
    
    rm,cm,norm = rm0.nestSpatialGrids(cm0,norm0)
    bias = cm-rm
    bias.ds[bias.varname] = np.abs(bias.ds[bias.varname])
    err = bias/norm

    
class Confrontation(object):

    def __init__(self,**kwargs):
        """
        source
        variable
        """
        self.source   = kwargs.get(  "source",None)
        self.variable = kwargs.get("variable",None)
        assert self.source is not None
        if not os.path.isfile(self.source):
            msg = "Cannot find the source, tried looking here: '%s'" % self.source
            raise ValueError(msg)

    def stageData(self,m):
        """
        
        """
        r = Variable(filename = self.source,
                     varname  = self.variable)
        t0,tf = r.timeBounds()
        c = m.getVariable(self.variable,t0=t0,tf=tf)
        c.convert(r.units())
        r,c = adjust_lon(r,c)
        return r,c
    
    def confront(self,m):
        """

        """
        r0,c0 = self.stageData(m)
        #r,c = r0.nestSpatialGrids(c0)
        ScoreBias(r0,c0,regions=['global','nhsa'])
        
        
if __name__ == "__main__":
    from ModelResult import ModelResult

    m = ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2",name="CESM2")
    m.findFiles()
    m.getGridInformation()
    #c = Confrontation(source = "/home/nate/data/ILAMB/DATA/gpp/FLUXCOM/tmp.nc",
    #                  variable = "gpp")
    c = Confrontation(source = "/home/nate/work/ILAMB-Data/CLASS/pr.nc",
                      variable = "pr")
    c.confront(m)
    
