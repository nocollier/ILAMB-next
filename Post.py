import os
import xarray as xr
import pandas as pd
import numpy as np
from Variable import Variable
from Regions import Regions

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

def is_site(da):
    """Do we have a 1D array of sites to plot?

    """
    if da.ndim != 1: return False
    if da.dims[0].startswith("time"): return False
    if ("lat" in list(da.coords) and "lon" in list(da.coords)): return True
    return True

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
    filename = []; varname = []; source = []; istime = []; isspace = []; issite = []
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
            issite  .append(is_site(da))
            region  .append(get_region(da))
            analysis.append(get_analysis(da))
            longname.append(get_longname(da))
            clr = (np.asarray(ds.color)*256).astype(int)
            clr = 'rgb(%d,%d,%d)' % (clr[0],clr[1],clr[2])
            colormap.append(clr if (istime[-1] and not isspace[-1]) else get_colormap(da,cmap))
    df = {"Filename":filename,"Variable":varname,"Model":source,"IsTime":istime,"IsSpace":isspace,
          "IsSite":issite,"Region":region,"Analysis":analysis,"Colormap":colormap,"Longname":longname}
    df = pd.DataFrame(df,columns=["Filename","Variable","Model","IsTime","IsSpace","IsSite","Region",
                                  "Analysis","Colormap","Longname"])
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
};""" % (pname,mname.replace("-","").replace(".",""),rname,x,y,mname,cname)
        return trace

    traces = ""
    plotnames = []
    plottypes = []
    for (plotname,region),dfg in df[df.IsTime & ~df.IsSpace].groupby(["Plot Name","Region"]):
        for i,r in dfg.iterrows():
            traces += _genTrace(r['Filename'],r['Variable'],r['Model'],r['Region'],
                                r['Plot Name'],r['Colormap'])
            plotnames.append("%s_%s_%s" % (r['Plot Name'],r['Model'].replace("-","").replace(".",""),r['Region']))
            if r['Plot Name'] not in plottypes: plottypes.append(r['Plot Name'])
    return traces,plotnames,plottypes
    
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
    df['ScalarName'] += " [" + df['Units'].astype(str) + "]"
    out = []
    for i,r in df.iterrows():
        d = dict(id=i)
        d.update(dict(r))
        out.append(d)
    out = str(out).replace(" nan"," NaN")
    return out

def generate_model_pulldown(df):
    """
    """
    models = sorted(list(df['Model'].dropna().unique()),key=lambda a: a.lower())
    models = [m for m in models if m != "Reference"]
    txt = [' '*14 + '<option value="%s">%s</option>' % (m,m) for m in models]
    txt[0] = txt[0].replace("option ","option selected ")
    entries = "\n".join(txt)
    html = """
 	    <h6 class="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
              <span>Model</span>
            </h6>
	    <select id="SelectModel" class="form-select" aria-label="ModelSelect" onclick="UpdateImages()">
%s
	    </select>""" % entries
    return html

def generate_region_pulldown(df):
    """
    """
    ilamb_regions = Regions()
    regions = list(df['Region'].dropna().unique())
    rnames = ["All Data" if r == 'None' else ilamb_regions.getRegionName(r) for r in regions]
    tmp = sorted(zip(regions,rnames),key=lambda x: '' if x[0]=='None' else x[1])
    regions = [r for r,_ in tmp]
    rnames  = [r for _,r in tmp]
    txt = [' '*14 + '<option value="%s">%s</option>' % (r,n) for r,n in zip(regions,rnames)]
    txt[0] = txt[0].replace("option ","option selected ")
    entries = "\n".join(txt)
    html = """
 	    <h6 class="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
              <span>Region</span>
            </h6>
	    <select id="SelectRegion" class="form-select" aria-label="RegionSelect" onclick="UpdateImages()">
%s
	    </select>""" % entries
    return html

def generate_analysis_menus(df):
    """
    """
    analyses = list(df['Analysis'].unique())
    aids = [a for a in analyses]
    html = """
              <h6 class="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
		<span>Analysis Types</span>
              </h6>
	      <ul class="nav flex-column mb-2">
		<li class="nav-item">
		  <a class="nav-link" id="Overview" aria-current="page" href="#" onclick="clickOverview()">
		    All
		  </a>
		</li>"""
    for aid,a in zip(aids,analyses):
        html += """
		<li class="nav-item">
		  <a class="nav-link" id="%s" href="#" onclick="click%s()">
		    %s
		  </a>
		</li>""" % (aid,a.replace(" ",""),a)  
    html += """
              </ul>"""
    return html

def generate_data_information(ref_file):
    try:
        ds = xr.open_dataset(ref_file)
    except:
        return ""
    html = """
              <h6 class="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
		<span>Data Information</span>
              </h6>
	      <ul class="nav flex-column mb-2">"""
    for head in ['title','institutions','version']:
        if head in ds.attrs:
            html += """
		<li class="nav-item">
		  <a class="nav-link">
		    <div class="fw-bold">%s</div>
		    %s
		  </a>
		</li>""" % (head.capitalize(),ds.attrs[head])
    html += """
	      </ul>"""
    return html

def generate_navigation_bar(df,ref_file):
    html = """
	<nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">	  
	  <div class="position-sticky pt-3">
%s
%s
	    <h6 class="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
              <span>Mode</span>
            </h6>
            <ul class="nav flex-column">
              <li class="nav-item">
		<a class="nav-link active" id="single" aria-current="page" href="#" onclick="clickOverview()">
		  Single Model (All Plots)
		</a>
              </li>    
              <li class="nav-item">
		<a class="nav-link" id="AllModels" aria-current="page" href="#" onclick="clickAllModels()">
		  All Models (By Plot)
		</a>
              </li>
%s      
%s	      
	  </div>
	</nav>\n""" % (generate_model_pulldown(df),
                       generate_region_pulldown(df),
                       generate_analysis_menus(df),
                       generate_data_information(ref_file))
    return html

def generate_main(df):
    analyses = list(df['Analysis'].unique())
    models = list(df['Model'].unique())
    if "Reference" in models: models.pop(models.index("Reference"))
    models = sorted(models,key=lambda x:x.lower())
    html = """
	<main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">	  
	  <div id="divTable">
	    <br><h2>Scalar Table</h2>
	    <div id="scalartable"></div>
	  </div>"""
    for a in analyses:
        dfa = df[df['Analysis']==a]
        plots = list(dfa['Plot Name'].unique())
        html += """
	  <div id="div%s">
	    <br><h2>%s</h2>""" % (a,a)
        for p in plots:
            plot_added = False
            for m in ['Reference',models[0]]:
                dfpm = dfa[(dfa['Plot Name']==p) & (dfa['Model']==m)]
                if len(dfpm)==0:
                    continue
                elif len(dfpm)==1:
                    plot_added = True
                    iid = "%s_%s" % (p,"Model" if m == models[0] else "Reference")
                    html += """
                <img id="%s" src="%s_None_%s.png" width="49%%">""" % (iid,m,p)
            dfp = dfa[(dfa['Plot Name']==p)]
            if not plot_added and len(dfp)>0:
                if dfp.iloc[0]['IsTime']:
                    html += """
	        <div id="%s" style="width:100%%;height:500px;"></div>""" % p
        html += """
          </div>"""

    dfp = df[df.IsTime==False]
    plots = list(dfp['Plot Name'].unique())
    html += """
	  <div id="divAllModels">	  
	    <br><h2>All Models</h2>
	    <select class="form-select" id="SelectPlot" aria-label="PlotSelect" onclick="UpdateImages()">"""
    
    txt = [' '*14 + '<option value="%s">%s</option>' % (p,dfp[dfp['Plot Name']==p].iloc[0]['Longname']) for p in plots]
    txt[0] = txt[0].replace("option ","option selected ")
    entries = "\n".join(txt)
    html += "\n%s" % entries
    html += """
            </select>"""

    txt = [' '*12 + '<img id="%s" src="%s_None_%s.png" width="32%%">' % (m,m,plots[0]) for m in models]
    entries = "\n" + "\n".join(txt)
    html += entries
    html += """	  
	</main>\n"""
    return html

def generate_analysis_toggles(dfs):
    analyses = [a for a in list(dfs['Analysis'].dropna().unique())]
    html = """
      function setActive(eid) {
        var atypes = %s;
	atypes.forEach((x, i) => document.getElementById(x).classList.remove('active'));
	document.getElementById("single").classList.remove('active');
	document.getElementById("AllModels").classList.remove('active');
	if (eid == "AllModels") {
	  document.getElementById("AllModels").classList.add('active');
	}else{
	  document.getElementById("single").classList.add('active');
	  document.getElementById(eid).classList.add('active');
	}
      };""" % (str(['Overview',]+analyses))
    funcs = ['Overview',] + analyses + ['AllModels',]
    for f in funcs:
        html +=  """
      function click%s() {
        setActive("%s");
	document.getElementById("divTable").style.display = "%s";""" % (f.replace(" ",""),f,"none" if f == "AllModels" else "block")
        if f != "AllModels":
            html += """
        var rsel  = document.getElementById("SelectRegion");
        var RNAME = rsel.options[rsel.selectedIndex].value;
        SetTable(RNAME,%s);""" % ("null" if f == "Overview" else '"%s"' % f)
        for a in analyses + ['AllModels']:
            style = "block" if f == a else "none"
            if f == "Overview" and a != "AllModels": style = "block"
            html += """
        document.getElementById("div%s").style.display = "%s";""" % (a,style)
        html += """
      };"""
    html += """
      clickOverview();"""

    return html

def generate_image_update(dfp):
    models = list(dfp['Model'].unique())
    if "Reference" in models: models.pop(models.index("Reference"))
    ref_plots = dfp[(dfp.Model=="Reference") & (dfp.IsTime==False)]['Plot Name'].unique()
    mod_plots = dfp[                           (dfp.IsTime==False)]['Plot Name'].unique()
    path = os.path.join(os.path.dirname(dfp['Filename'].iloc[0]))
    html = """
      function UpdateImages() {
        var rsel  = document.getElementById("SelectRegion");
        var RNAME = rsel.options[rsel.selectedIndex].value;
        var msel  = document.getElementById("SelectModel");
        var MNAME = msel.options[msel.selectedIndex].value;
        var psel  = document.getElementById("SelectPlot");
        var PNAME = psel.options[psel.selectedIndex].value;
        var path = '%s';""" % path
    html += """
        var ref_plots = [%s];
        ref_plots.forEach((x, i) => document.getElementById(x + '_Reference').src = 'Reference_' + RNAME + '_' + x + '.png');""" % (", ".join(['"%s"' % p for p in ref_plots]))
    html += """
        var mod_plots = [%s];
        mod_plots.forEach((x, i) => document.getElementById(x + '_Model').src = MNAME + '_' + RNAME + '_' + x + '.png');""" % (", ".join(['"%s"' % p for p in mod_plots]))
    html += """
        var models = [%s];
        models.forEach((x, i) => document.getElementById(x).src =  x + '_' + RNAME + '_' + PNAME + '.png');""" % (", ".join(['"%s"' % m for m in models]))
    html += """
      };"""
    return html

def generate_script(dfp,dfs):
    html = """
    <script>
      %s

      var tableData = %s;
      function SetTable(region,analysis) {
	  
	  /* creates a nested dictionary of models and scalar names */
	  var cols = ['Model'];
	  var o = tableData.reduce( (a,b) => {
	      a[b.Model] = a[b.Model] || {};
	      if (b.Region != region) return a;
	      if (analysis) {
		  if (b.Analysis != analysis) return a;
	      };
	      a[b.Model][b.ScalarName] = b.Data;
	      if (cols.indexOf(b.ScalarName) < 0) cols.push(b.ScalarName);
	      return a;
	  }, {});
	  
	  /* build columns dictionary for the table */
	  var cols = cols.map(function(k) {
	      return {title:k,field:k};
	  });
	  
	  /* unnest the dictionary to put it how the tabulator wants it */
	  var data = Object.keys(o).map(function(k) {
	      return Object.assign({'Model':k},o[k]);
	  });
	  
	  var table = new Tabulator("#scalartable", {
	      data:data,
	      layout:"fitData",
	      columns:cols
	  });
      };\n""" % (generate_image_update(dfp),convert_scalars_to_str(dfs))
    html += generate_analysis_toggles(dfs)
    code,names,types = generate_jsplotly_curves(dfp)
    html += code

    html += """
      var layout = {
    	  font: { size: 24 }
      };"""
    for t in types:
        n = [a for a in names if a.startswith(t) and a.endswith("None")]
        html += """
      Plotly.newPlot('%s',[%s],layout);""" % (t,",".join(n))
        
    html += """
    </script>"""
    return html

def generate_dataset_html(dfp,dfs,ref_file):
    
    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>gpp | FLUXCOM</title>
    <link href="https://unpkg.com/tabulator-tables@4.0.5/dist/css/tabulator.min.css" rel="stylesheet">
    <script type="text/javascript" src="https://unpkg.com/tabulator-tables@4.0.5/dist/js/tabulator.min.js"></script>        
    <link href="bootstrap.min.css" rel="stylesheet">
    <link href="dashboard.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.4.2.min.js"></script>
  </head>
  <body>
    <header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
      <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3" href="#">gpp | FLUXCOM | 1980-2013</a>
      <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false" aria-label="Toggle navigation">
	<span class="navbar-toggler-icon"></span>
      </button>
    </header>
    <div class="container-fluid">
      <div class="row">\n"""
    html += generate_navigation_bar(dfp,ref_file)
    html += generate_main(dfp)
    html += """
      </div>
    </div>"""
    html += generate_script(dfp,dfs)
    html += """
  </body>
</html>"""
    return html

    
if __name__ == "__main__":
    import glob
    src = "FLUXCOM"
    dfp = generate_plot_database(glob.glob(f"_test/gpp/{src}/*.nc"),cmap="Greens")
    dfs = generate_scalar_database(glob.glob(f"_test/gpp/{src}/*.csv"))
    print(generate_dataset_html(dfp,dfs,f"~/data/ILAMB/DATA/gpp/{src}/gpp.nc"))


