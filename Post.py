import xarray as xr
import pandas as pd
import numpy as np

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

def get_colormap(da,cmap=None):
    """Colormaps can be encoded or use defaults.

    """
    if da.dims[0].startswith("time"): return pd.NA
    if "cmap"  in da.attrs: return da.attrs["cmap"]
    if "score" in da.name:  return "winter"
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
    region = []; analysis = []; colormap = []
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
            colormap.append(get_colormap(da,cmap))
    df = {"Filename":filename,"Variable":varname,"Model":source,"IsTime":istime,"IsSpace":isspace,
          "Region":region,"Analysis":analysis,"Colormap":colormap}
    df = pd.DataFrame(df,columns=["Filename","Variable","Model","IsTime","IsSpace","Region","Analysis","Colormap"])
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
    
def generate_scalar_database(csvfiles):
    """Build a single pandas dataframe with all scalar information.

    """
    df = [pd.read_csv(f) for f in csvfiles]
    df = pd.concat(df,ignore_index=True).drop_duplicates(['Model','Region','ScalarName'])
    return df
    
if __name__ == "__main__":
    import glob
    dfp = generate_plot_database(glob.glob("_build/gpp/FLUXCOM/*.nc"),cmap="Greens")
    dfs = generate_scalar_database(glob.glob("_build/gpp/FLUXCOM/*.csv"))

    

    
