from Variable import ilamb_variable
from ILAMB import ilamblib as il
from netCDF4 import Dataset
from sympy import sympify
import xarray as xr
import numpy as np
import os
import re

def MultiModelMean(V):
    models = list(V.keys())
    # make sure all variables are comparable to each other
    v0 = V[models[0]]
    for a in models:
        for b in models:
            V[a],V[b] = il.MakeComparable(V[a],V[b])
    # naive mean when everything is the same shape
    data_sum  = np.zeros(V[models[0]].data.shape)
    data_sum2 = np.zeros(V[models[0]].data.shape)
    data_cnt  = np.zeros(V[models[0]].data.shape,dtype=int)
    with np.errstate(all='ignore'):
        for a in models:
            data       =  V[a].data.data
            count      = (V[a].data.mask==False)
            data[count==False] = 0
            data_sum  += data     
            data_sum2 += data*data
            data_cnt  += count
        mean = data_sum  / data_cnt.clip(1)
        std  = np.sqrt((data_sum2 / data_cnt.clip(1) - mean*mean).clip(0))
    mean = np.ma.masked_array(mean,mask=(data_cnt==0))
    bnd = np.ma.masked_array(np.zeros(mean.shape + (2,)))
    bnd[...,0] = mean-std
    bnd[...,1] = mean+std
    return Variable(name = v0.name,
                    unit = v0.unit,
                    data = mean,    data_bnds = bnd,
                    time = v0.time, time_bnds = v0.time_bnds,
                    lat  = v0.lat,  lat_bnds  = v0.lat_bnds,
                    lon  = v0.lon,  lon_bnds  = v0.lon_bnds)
        
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
        for child in self.children:
            s += "  + %s\n" % (child)
        return s

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
                    with Dataset(path) as dset:
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
            atm = self._getVariableChild("areacella",synonyms=["area"]).ilamb.convert("m2")
        except:
            pass
        try:
            ocn = self._getVariableChild("areacello").ilamb.convert("m2")
        except:
            pass
        try:
            lnd = self._getVariableChild("sftlf",synonyms=["landfrac"]).ilamb.convert("1")
        except:
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
            vmean = MultiModelMean(v)
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
        V = self.variables[vname]
        if len(V) == 0:
            raise ValueError("No %s file available in %s" % (vname,self.name))
        elif len(V) > 1:
            dset = xr.concat([xr.open_dataset(f) for f in V],dim="time")
        else:
            dset = xr.open_dataset(V[0])
        v = dset[vname]
        v.ilamb.setBounds(dset)

        area = frac = None
        if "cell_measures" in v.attrs and vname not in ["areacella","areacello"]:
            if "areacella" in v.attrs["cell_measures"]: area = self.area_atm
            if "areacello" in v.attrs["cell_measures"]: area = self.area_ocn
            if "cell_methods" in v.attrs:
                if "where land" in v.attrs["cell_methods"]: frac = self.frac_lnd
            v.ilamb.setMeasure(area=area,fraction=frac)
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
