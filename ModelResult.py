from Variable import Variable
from sympy import sympify
import xarray as xr
import numpy as np
import os
import re

def compute_multimodel_mean(V):
    """
    for now require spatial grids are fixed
    assume we have spatio-temporal data
    """
    tb = [V[v].timeBounds() for v in V]
    t0 = max([t[0] for t in tb])
    tf = min([t[1] for t in tb])
    for v in V: V[v].ds = V[v].ds.sel(time=slice(t0,tf))
    for a in V:
        A = V[a]
        for b in V:
            B = V[b]
            assert A.ds[A.varname].shape == B.ds[B.varname].shape
    ds = xr.concat([V[v].ds for v in V],dim='model').mean(dim='model')
    mn = Variable(da=ds[A.varname],varname=A.varname,
                  cell_measure=A.ds['cell_measure'],time_measure=A.ds['time_measure'])
    mn.ds[mn.varname].attrs = A.ds[A.varname].attrs
    return mn

def compute_time_measure(ds):
    """
    this is duplicate code from inside Variable
    """
    dt = None
    tb_name = None
    if "bounds" in ds['time'].attrs: tb_name = ds['time'].attrs['bounds']
    if tb_name is not None and tb_name in ds:
        dt = ds[tb_name]
        nb = dt.dims[-1]
        dt = dt.diff(nb).squeeze()
        dt *= 1e-9/86400 # [ns] to [d]
        dt  = dt.astype('float')
    return dt

class ModelResult():
    """
    name
    """
    def __init__(self,path,**kwargs):

        # 
        self.name = kwargs.get("name","model")
        self.color = (0,0,0)

        # A 'model' could be a collection of results, as in an
        # ensemble or a MIP. The collection of results could also be a
        # perturbation study.
        self.parent = None
        self.children = {}

        # What is the top level directory where files are located?
        self.path = path

        # What variables do we have and what files are they in?
        self.variables = {}

        # Synonyms
        self.synonyms = {}

        # Areas
        self.area_atm = self.area_ocn = self.frac_lnd = None

    def __str__(self):
        s  = ""
        s += "ModelResult: %s\n" % self.name
        s += "-"*(len(self.name)+13) + "\n"
        if self.children:
            for child in self.children:
                s += "  + %s\n" % (child)
        else:
            for v in sorted(self.variables,key=lambda k: k.lower()):
                s += "  + %s\n" % (v)                
        return s

    def __repr__(self):
        return self.__str__()
    
    def _byRegex(self,group_regex):
        """
        """
        groups = {}
        for v in self.variables.keys():
            to_remove = []
            for f in self.variables[v]:
                m = re.search(group_regex,f)
                if m:
                    g = m.group(1)
                    if g not in groups: groups[g] = {}
                    if v not in groups[g]: groups[g][v] = []
                    groups[g][v].append(f)
                    to_remove.append(f)
            for r in to_remove: self.variables[v].remove(r) # get rid of those we passed to models
        return groups

    def _byAttr(self,group_attr):
        """
        """
        groups = {}
        for v in self.variables.keys():
            to_remove = []
            for f in self.variables[v]:
                with Dataset(f) as dset:
                    if group_attr in dset.ncattrs():
                        g = dset.getncattr(group_attr)
                        if g not in groups: groups[g] = {}
                        if v not in groups[g]: groups[g][v] = []
                        groups[g][v].append(f)
                        to_remove.append(f)
            for r in to_remove: self.variables[v].remove(r) # get rid of those we passed to models
        return groups

    def findFiles(self,**kwargs):
        """
        file_paths
        group_regex
        group_attr
        """
        file_paths = kwargs.get("file_paths",[self.path])
        for file_path in file_paths:
            if file_path is None: continue
            for root,dirs,files in os.walk(file_path,followlinks=True):
                for f in files:
                    if not f.endswith(".nc"): continue
                    path = os.path.join(root,f)
                    with xr.open_dataset(path) as dset:
                        for key in dset.variables.keys():
                            if key not in self.variables: self.variables[key] = []
                            self.variables[key].append(path)

        # create sub-models automatically in different ways
        group_regex = kwargs.get("group_regex",None)
        group_attr  = kwargs.get("group_attr" ,None)
        groups      = {}
        if not groups and group_regex is not None: groups = self._byRegex(group_regex)
        if not groups and group_attr  is not None: groups = self._byAttr (group_attr )
        for g in groups:
            m = ModelResult(self.path,name=g)
            m.variables = groups[g]
            m.parent = self
            self.children[m.name] = m
        return self

    def getGridInformation(self):
        """
        """
        atm = ocn = lnd = None
        try:
            atm = self._getVariableChild("areacella",synonyms=["area"]).convert("m2")
            atm = atm.ds[atm.varname]
        except Exception as e:
            pass
        try:
            ocn = self._getVariableChild("areacello").convert("m2")
            ocn = ocn.ds[ocn.varname]
        except Exception as e:
            pass
        try:
            lnd = self._getVariableChild("sftlf",synonyms=["landfrac"]).convert("1")
            lnd = lnd.ds[lnd.varname]
        except Exception as e:
            pass
        if atm is not None: self.area_atm = atm
        if ocn is not None: self.area_ocn = ocn
        if lnd is not None: self.frac_lnd = lnd
        for child in self.children: self.children[child].getGridInformation()

    def addModel(self,m):
        """
        m : ILAMB.ModelResult or list of ILAMB.ModelResult
          the model(s) to add as children
        """
        if type(m) == type(self):
            if m.name not in self.children:
                self.children[m.name] = m
                m.parent = self
        elif type(m) == type([]):
            for mm in m:
                if type(mm) != type(self): continue
                if mm.name not in self.children:
                    self.children[mm.name] = mm
                    mm.parent = self

    def addSynonym(self,expr):
        """

        """
        assert type(expr) == type("")
        if expr.count("=") != 1: raise ValueError("Add a synonym by providing a string of the form 'a = b + c'")
        key,expr = expr.split("=")
        if key not in self.synonyms: self.synonyms[key] = []
        # check that the free symbols of the expression are variables
        for arg in sympify(expr).free_symbols:
            assert arg.name in self.variables
        self.synonyms[key].append(expr)

    def getVariable(self,vname,**kwargs):
        """
        synonyms
        mean
        """
        mean = kwargs.get("mean",False)
        if not self.children: return self._getVariableChild(vname,**kwargs)
        v = {}
        for child in self.children:
            v[child] = self.children[child]._getVariableChild(vname,**kwargs)
        if mean:
            vmean = compute_multimodel_mean(v)
            return v,vmean
        return v

    def _getVariableChild(self,vname,**kwargs):
        """
        synonymns
        """
        # If a model has synonyms defined, these take
        # prescendence. But we may need to look for they using the
        # synonyms provided in the 'getVariable' call.
        synonyms = kwargs.get("synonyms",[])
        if type(synonyms) != type([]): synonyms = [synonyms]
        synonyms = [vname,] + synonyms
        Ss = [s for s in synonyms if s in self.synonyms ]
        Vs = [s for s in synonyms if s in self.variables]
        if Ss: return self._getVariableExpression(vname,self.synonyms[Ss[0]][0],**kwargs)
        if Vs: return self._getVariableNoSyn(Vs[0],**kwargs)
        raise ValueError("Cannot find '%s' in '%s'" % (vname,self.name))

    def _getVariableNoSyn(self,vname,**kwargs):
        """At some point we need a routine to get the data from the model
        files without the possibility of synonyms or we end up in an
        infinite recursion. This is where we should do all the
        trimming / checking of sites, etc.
        """
        # Even if multifile, we need to peak at attributes from one file
        V = sorted(self.variables[vname])
        if len(V) == 0:
            raise ValueError("No %s file available in %s" % (vname,self.name))
        else:
            dset = xr.open_dataset(V[0])
        v = dset[vname]
            
        # Scan attributes for cell measure information
        area = None
        if "cell_measures" in v.attrs and vname not in ["areacella","areacello"]:
            if ("areacella" in v.attrs["cell_measures"] and
                self.area_atm is not None): area = self.area_atm.copy()
            if ("areacello" in v.attrs["cell_measures"] and
                self.area_ocn is not None): area = self.area_ocn.copy()
            if "cell_methods" in v.attrs and area is not None:
                if ("where land" in v.attrs["cell_methods"] and
                    self.frac_lnd is not None): area *= self.frac_lnd

        t0 = kwargs.get("t0",None)
        tf = kwargs.get("tf",None)
        if len(V) == 1:
            v = Variable(filename=V[0],varname=vname,cell_measure=area,t0=t0,tf=tf)
        else:
            V = [xr.open_dataset(v).sel(time=slice(t0,tf)) for v in V]
            V = sorted([v for v in V if v.time.size>0],key=lambda a:a.time[0])
            try:
                ds = xr.concat(V,dim='time')
            except:
                msg = "Could not concat variable '%s' from files:\n  %s" % (vname,"\n  ".join(sorted(self.variables[vname])))
                raise ValueError(msg)
            v = Variable(da=ds[vname],varname=vname,cell_measure=area,time_measure=compute_time_measure(ds))
            
        latlon = kwargs.get("latlon",None)
        if latlon is not None:
            raise ValueError("Datasite extraction not yet implemented")
            lat,lon = latlon
            if v.lon.data.max() > 180:
                if lon < 0: lon += 360
            v = v.sel(lat=lat,lon=lon,method='nearest')
            
        return v

    def _getVariableExpression(self,vname,expr,**kwargs):
        """
        """
        def _checkDim(dimref,dim):
            if dimref is None: dimref = dim
            if dim is not None: assert(np.allclose(dimref,dim))
            return dimref
        t = tb = x = xb = y = yb = d = db = n = A = mask = None
        data = {}; unit = {}
        for arg in sympify(expr).free_symbols:
            v  = self._getVariableNoSyn(arg.name,**kwargs)
            t  = _checkDim(t ,v.time)
            tb = _checkDim(tb,v.time_bnds)
            x  = _checkDim(x ,v.lon)
            xb = _checkDim(xb,v.lon_bnds)
            y  = _checkDim(y ,v.lat)
            yb = _checkDim(yb,v.lat_bnds)
            d  = _checkDim(d ,v.depth)
            db = _checkDim(db,v.depth_bnds)
            n  = _checkDim(n ,v.ndata)
            A  = _checkDim(A ,v.area)
            data[arg.name] = v.data.data
            unit[arg.name] = v.unit
            if mask is None: mask = v.data.mask
            mask += v.data.mask
        with np.errstate(all='ignore'):
            result,unit = il.SympifyWithArgsUnits(expr,data,unit)
        data = np.ma.masked_array(np.nan_to_num(result),mask=mask+np.isnan(result))
        return Variable(name = vname, unit = unit, data = data, ndata = n, area = A,
                        time  = t, time_bnds  = tb,
                        lat   = y, lat_bnds   = yb,
                        lon   = x, lon_bnds   = xb,
                        depth = d, depth_bnds = db)
