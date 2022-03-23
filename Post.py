import xarray as xr
import pandas as pd
import numpy as np
from Variable import Variable

def is_time(da):
    """Do we have a 1D temporal curve to plot?

    """
    if da.ndim != 1: return False
    if da.dims[0].startswith("time"): return True
    return False

def is_space(da):
    """Do we have a 2D map to plot?

    """
    if  da.ndim != 2: return False
    if (da.dims[0].startswith("lat") and da.dims[1].startswith("lon")): return True
    return False

def get_region(da):
    """Regions can be encoded in the name or attributes.

    """
    region = pd.NA
    if "_over_" in da.name: region = da.name.split("_over_")[-1]
    if "region" in da.attrs: region = da.attrs['region']
    return region

def get_analysis(da):
    """The analysis name can be encoded in the attributes.

    """
    analysis = pd.NA
    if "analysis" in da.attrs: analysis = da.attrs['analysis']
    return analysis

def get_longname(da):
    """The plot name can be encoded in the attributes.

    """
    name = pd.NA
    if "longname" in da.attrs: name = da.attrs['longname']
    return name

def get_colormap(da,cmap=None):
    """Colormaps can be encoded or use defaults.

    """
    if "cmap"  in da.attrs: return da.attrs["cmap"]
    if "score" in da.name:  return "plasma"
    if "bias"  in da.name:  return "seismic"
    if "rmse"  in da.name:  return "Oranges"
    if "shift" in da.name:  return "PRGn"
    if "phase" in da.name:  return "jet"
    if  cmap   is None:     return "viridis"
    return cmap

def generate_plot_database(ncfiles,cmap=None):
    """Build a pandas dataframe with information for plotting.

    """
    if type(ncfiles) is not list: ncfiles = [ncfiles]
    filename = []; varname = []; source = []; istime = []; isspace = []
    region = []; analysis = []; colormap = []; longname = []
    for f in ncfiles:
        ds = xr.open_dataset(f)
        for v in ds:
            da = ds[v]
            filename.append(f)
            varname .append(v)
            source  .append(ds.name)
            istime  .append(is_time(da))
            isspace .append(is_space(da))
            region  .append(get_region(da))
            analysis.append(get_analysis(da))
            longname.append(get_longname(da))
            clr = (np.asarray(ds.color)*256).astype(int)
            clr = 'rgb(%d,%d,%d)' % (clr[0],clr[1],clr[2])
            colormap.append(clr if (istime[-1] and not isspace[-1]) else get_colormap(da,cmap))
    df = {"Filename":filename,"Variable":varname,"Model":source,"IsTime":istime,"IsSpace":isspace,
          "Region":region,"Analysis":analysis,"Colormap":colormap,"Longname":longname}
    df = pd.DataFrame(df,columns=["Filename","Variable","Model","IsTime","IsSpace","Region","Analysis","Colormap",
                                  "Longname"])
    df = find_plot_limits(df)
    df['Plot Name'] = [p[0] for p in df.Variable.str.split("_")]
    return df

def find_plot_limits(df,percentile=[1,99]):
    """Add the plotting limits to the plot database.

    If the variable name has 'score' in it, the limits are set to
    [0,1]. Otherwise, we create an array of all variable data across
    all sources and then set the plot limits to a percentile.

    """
    vmin = {}; vmax = {}
    for v in df.Variable.unique():
        if "score" in v:
            vmin[v] = 0
            vmax[v] = 1
        else:
            values = []
            for fname in df[df.Variable==v].Filename:
                with xr.load_dataset(fname) as ds:
                    da = ds[v]
                    a = da.stack(dim=da.dims)
                    values.append(a.where(np.isfinite(a),drop=True).values)
            values = np.hstack(values)
            values = np.percentile(values,percentile)
            if v.startswith("bias_") or v.startswith("shift_"):
                values[...] = np.abs(values).max()
                values[0]  *= -1
            vmin[v] = values[0]
            vmax[v] = values[1]
    vmin = [vmin[v] for v in df.Variable]
    vmax = [vmax[v] for v in df.Variable]
    df['Plot Min'] = vmin
    df['Plot Max'] = vmax
    return df

def generate_jsplotly_curves(df):
    """

    """
    def _genTrace(filename,varname,mname,rname,pname,cname):
        v = Variable(filename=filename,varname=varname)
        t = v.ds[v.ds[v.varname].dims[0]]
        if np.issubdtype(t,float):
            x = ",".join([str(float(i)) for i in t])
        else:
            x = ",".join(['"%s"' % i.values for i in t.dt.strftime("%Y-%m-%d")])
        da = v.ds[v.varname]
        y = ",".join(["%g" % t for t in da])
        trace = """        
var %s_%s_%s = {
  x: [%s],
  y: [%s],
  mode: 'lines',
  name: '%s',
  line: { color: '%s' }
};""" % (pname,mname,rname,x,y,mname,cname)
        return trace

    traces = ""
    for (plotname,region),dfg in df[df.IsTime & ~df.IsSpace].groupby(["Plot Name","Region"]):
        for i,r in dfg.iterrows():
            traces += _genTrace(r['Filename'],r['Variable'],r['Model'],r['Region'],
                                r['Plot Name'],r['Colormap'])
    return traces
    
def generate_scalar_database(csvfiles):
    """Build a single pandas dataframe with all scalar information.

    """
    df = [pd.read_csv(f) for f in csvfiles]
    df = pd.concat(df,ignore_index=True).drop_duplicates(['Model','Region','ScalarName'])    
    return df

def convert_scalars_to_str(dfs):
    """
    # determine preferred column order
    analyses = df.Analysis.unique()
    columns = []
    for stype in ['scalar','score']:
        for a in analyses:
            columns += list(df[(df.Analysis==a)&(df.ScalarType==stype)].ScalarName.unique())
    columns += ['Overall Score']
    """
    df = dfs.copy()
    df['Data'] = df['Data'].apply('{:,.3g}'.format)
    df['ScalarName'] += " [" + df['Units'] + "]"
    out = []
    for i,r in df.iterrows():
        d = dict(id=i)
        d.update(dict(r))
        out.append(d)
    out = str(out).replace(" nan"," NaN")
    return out
        
    
if __name__ == "__main__":
    import glob
    dfp = generate_plot_database(glob.glob("_build/gpp/FLUXCOM/*.nc"),cmap="Greens")
    #print(generate_jsplotly_curves(dfp))
    dfs = generate_scalar_database(glob.glob("_build/gpp/FLUXCOM/*.csv"))
    print(convert_scalars_to_str(dfs))
        
