import os
from netCDF4 import Dataset
import numpy as np
import xarray as xr

class Regions(object):
    """A class for unifying the treatment of regions in ILAMB.

    This class holds a list of all regions currently registered in the
    ILAMB system via a static property of the class. It also comes
    with methods for defining additional regions by lat/lon bounds or
    by a mask specified by a netCDF4 file. A set of regions used in
    the Global Fire Emissions Database (GFED) is included by default.

    """
    _regions = {}
    _sources = {}
    
    @property
    def regions(self):
        """Returns a list of region identifiers."""
        return Regions._regions.keys()
    
    def addRegionLatLonBounds(self,label,name,lats,lons,source="user-provided latlon bounds"):
        """Add a region by lat/lon bounds.

        Parameters
        ----------
        label : str
            the unique region identifier (lower case, no spaces or special characters)
        name : str
            the name of the region (as will appear in the HTML pull down menu)
        lats : array-like of size 2
            the minimum and maximum latitudes defining the region on the interval (-90,90)
        lons : array-like of size 2
            the minimum and maximum longitudes defining the region on the interval (-180,180)
        source : str, optional
            a string representing the source of the region, purely cosmetic
        """
        assert len(lats)==2
        assert len(lons)==2
        rtype = 0
        Regions._regions[label] = [rtype,name,lats,lons]
        Regions._sources[label] = source
        
    def addRegionNetCDF4(self,filename):
        """Add regions found in a netCDF4 file.

        This routine will search the target filename's variables for
        2-dimensional datasets which contain indices representing
        distinct non-overlapping regions. Each unique non-masked index
        found in this dataset will be added to the global list of
        regions along with a mask representing the region. The names
        of the regions are taken from a required attribute in the
        variable called 'labels'. This attribute should point to a
        variable which is a string array labeling each index found in
        the two-dimensional dataset.

        For example, the following header represents a dataset encoded
        to represent 50 of the world's largest river basins. The
        'basin_index' variable contains integer indices 0 through 49
        where index 0 is labeled by the 0th label found in the 'label'
        variable::

          dimensions:
                lat = 360 ;
                lon = 720 ;
                n = 50 ;
          variables:
                string label(n) ;
                        label:long_name = "basin labels" ;
                float lat(lat) ;
                        lat:long_name = "latitude" ;
                        lat:units = "degrees_north" ;
                float lon(lon) ;
                        lon:long_name = "longitude" ;
                        lon:units = "degrees_east" ;
                int basin_index(lat, lon) ;
                        basin_index:labels = "label" ;

        Parameters
        ----------
        filename : str
            the full path of the netCDF4 file containing the regions

        Returns
        -------
        regions : list of str
            a list of the keys of the regions added.
        """
        rtype = 1
        ds = xr.load_dataset(filename)
        labels = list(ds[ds['ids'].attrs['labels']].to_numpy())
        names  = list(ds[ds['ids'].attrs['names' ]].to_numpy())
        for label,name in zip(labels,names):
            da = xr.where(ds['ids']==labels.index(label),1,0)
            Regions._regions[label] = [rtype,name,da]
            Regions._sources[label] = os.path.basename(filename)
        return labels

    def getRegionName(self,label):
        """Given the region label, return the full name.

        Parameters
        ----------
        label : str
            the unique region identifier

        Returns
        -------
        name : str
            the long name of the region
        """
        return Regions._regions[label][1]
    
    def getRegionSource(self,label):
        """Given the region label, return the source.

        Parameters
        ----------
        label : str
            the unique region identifier

        Returns
        -------
        name : str
            the source of the region
        """
        return Regions._sources[label]

    def getMask(self,label,var):
        """Given the region label and a ILAMB.Variable, return a mask appropriate for that variable.

        Parameters
        ----------
        label : str
            the unique region identifier
        var : ILAMB.Variable.Variable
            the variable to which we would like to apply a mask

        Returns
        -------
        mask : numpy.ndarray
            a boolean array appropriate for masking the input variable data
        """
        rdata = Regions._regions[label]
        rtype = rdata[0]
        if rtype == 0:
            rtype,rname,rlat,rlon = rdata
            lat = var.ds[var.lat_name]
            lon = var.ds[var.lon_name]
            if lon.max() > 180: rlon = (np.asarray(rlon)+360)%360
            keep = [u for u in var.ds if var.lat_name in var.ds[u].dims and var.lon_name in var.ds[u].dims]
            ds = var.ds.drop_vars([u for u in var.ds if u not in keep])
            ds = xr.where((lat>=rlat[0])*(lat<=rlat[1])*(lon>=rlon[0])*(lon<=rlon[1]),ds,np.nan)
            return ds
        elif rtype == 1:
            rtype,rname,da = rdata
            out = da.interp(lat=var.ds[var.lat_name],lon=var.ds[var.lon_name],method='nearest')
            return out==False
        msg = "Region type #%d not recognized" % rtype
        raise ValueError(msg)


    def hasData(self,label,var):
        """Checks if the ILAMB.Variable has data on the given region.

        Parameters
        ----------
        label : str
            the unique region identifier
        var : ILAMB.Variable.Variable
            the variable to which we would like check for data

        Returns
        -------
        hasdata : boolean
            returns True if variable has data on the given region
        """
        axes = range(var.data.ndim)
        if var.spatial: axes = axes[:-2]
        if var.ndata  : axes = axes[:-1]
        keep = (self.getMask(label,var)==False)
        if var.data.mask.size == 1:
            if var.data.mask: keep *= 0
        else:
            keep *= (var.data.mask == False).any(axis=tuple(axes))
        if keep.sum() > 0: return True
        return False

if "global" not in Regions().regions:

    # Populate some regions
    r = Regions()
    src = "ILAMB internal"
    r.addRegionLatLonBounds("global","Globe",(-89.999, 89.999),(-179.999, 179.999),src)
    r.addRegionLatLonBounds("globe","Global - All",(-89.999, 89.999),(-179.999, 179.999),src)

    # GFED regions
    src = "Global Fire Emissions Database (GFED)"
    r.addRegionLatLonBounds("bona","Boreal North America",             ( 49.75, 79.75),(-170.25,- 60.25),src)
    r.addRegionLatLonBounds("tena","Temperate North America",          ( 30.25, 49.75),(-125.25,- 66.25),src)
    r.addRegionLatLonBounds("ceam","Central America",                  (  9.75, 30.25),(-115.25,- 80.25),src)
    r.addRegionLatLonBounds("nhsa","Northern Hemisphere South America",(  0.25, 12.75),(- 80.25,- 50.25),src)
    r.addRegionLatLonBounds("shsa","Southern Hemisphere South America",(-59.75,  0.25),(- 80.25,- 33.25),src)
    r.addRegionLatLonBounds("euro","Europe",                           ( 35.25, 70.25),(- 10.25,  30.25),src)
    r.addRegionLatLonBounds("mide","Middle East",                      ( 20.25, 40.25),(- 10.25,  60.25),src)
    r.addRegionLatLonBounds("nhaf","Northern Hemisphere Africa",       (  0.25, 20.25),(- 20.25,  45.25),src)
    r.addRegionLatLonBounds("shaf","Southern Hemisphere Africa",       (-34.75,  0.25),(  10.25,  45.25),src)
    r.addRegionLatLonBounds("boas","Boreal Asia",                      ( 54.75, 70.25),(  30.25, 179.75),src)
    r.addRegionLatLonBounds("ceas","Central Asia",                     ( 30.25, 54.75),(  30.25, 142.58),src)
    r.addRegionLatLonBounds("seas","Southeast Asia",                   (  5.25, 30.25),(  65.25, 120.25),src)
    r.addRegionLatLonBounds("eqas","Equatorial Asia",                  (-10.25, 10.25),(  99.75, 150.25),src)
    r.addRegionLatLonBounds("aust","Australia",                        (-41.25,-10.50),( 112.00, 154.00),src)
