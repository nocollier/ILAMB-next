import os
from Confrontation import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def critical_soil_moisture(m,**kwargs):
    """LoCo metric.

    https://github.com/pdirmeyer/l-a-cheat-sheets/blob/main/Coupling_metrics_V30_CriticalSM.pdf
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD031672#jgrd56114-tbl-0001
    
    """
    
    # Parse confrontation options
    t0      = kwargs.get("t0","1980-01-01")
    tf      = kwargs.get("tf","2000-01-01")
    nveg    = kwargs.get("veg","evspsbl")
    nwater  = kwargs.get("water","mrsos")
    nenergy = kwargs.get("energy","ts")
    path    = kwargs.get("path","./")
    aname   = "Critical Soil Moisture"
    
    # Get the model variables and process, if not available we just
    # return nothing
    try:
        veg    = m.getVariable(nveg   ,t0=t0,tf=tf)
        energy = m.getVariable(nenergy,t0=t0,tf=tf)
        water  = m.getVariable(nwater ,t0=t0,tf=tf)
    except:
        return {},{},[]
    for v in [veg,energy,water]: v.detrend().decycle()

    # Note: In the paper, they compute correlations per season, but
    # their metric was for Europe and ours is global. The notion of
    # seasons does not transfer very well. Need to think up something
    # else as correlations not relevant when vegetation is not active.
    R_water  =  water.correlation(veg,dim='time')
    R_energy = energy.correlation(veg,dim='time')
    R = R_energy - R_water
    R.setAttr("longname","Correlation Difference")
    R.setAttr("region","None")
    
    # Analysis includes looking at moisture at transitions of
    # correlations. So we read the water variable again and attempt to
    # convert to a volumetric %
    water = m.getVariable(nwater,t0=t0,tf=tf).integrate(mean=True,dim='time')
    if nwater.lower() == "mrsos":
        density = 1000. # [kg m-3]
        thick   = 0.1   # [m]
        water.convert("kg m-2")
        water.ds[water.varname] /= density*thick
        water.ds[water.varname].attrs['unit'] = "1"        
    df = pd.DataFrame({"R"       :       R.ds[       R.varname].values.flatten(),
                       "R_water" : R_water.ds[ R_water.varname].values.flatten(),
                       "R_energy":R_energy.ds[R_energy.varname].values.flatten(),
                       nwater    :   water.ds[   water.varname].values.flatten()})
    df = df.dropna().sort_values(nwater)

    # Moving least squares
    per0 = kwargs.get("per0",0.05)
    perf = kwargs.get("perf",0.95)
    wper = kwargs.get("wper",1/20)
    qn = df.quantile([per0,perf])
    x  = np.linspace(qn.iloc[0][nwater],qn.iloc[1][nwater],25)
    w  = (qn.iloc[1][nwater]-qn.iloc[0][nwater])*wper
    df_mls = pd.DataFrame([df[(df[nwater] > x0-w)*(df[nwater] < x0+w)].mean() for x0 in x])
    df_mls[nwater] = x

    # Insert a R=0 and interpolate to find CSM
    df_mls.loc[-1, 'R'] = 0
    df_mls = df_mls.sort_values('R').interpolate()
    df_mls = df_mls.sort_values(nwater).reset_index().drop(columns='index')
    csm = df_mls[(df_mls['R'].abs())<1e-6]

    # Fraction of points less than CSM
    fless = (df[nwater] < csm.iloc[0][nwater]).sum()/len(df[nwater])
    
    # Critical Soil Moisture Scatter Plot
    fig,ax = plt.subplots(figsize=(6,5),tight_layout=True,dpi=200)
    ax.scatter(df[nwater],df['R'],lw=0,color='0.8',alpha=0.1)
    ax.plot(df_mls[nwater],df_mls['R_energy'],'-r')
    ax.plot(df_mls[nwater],df_mls['R_water'],'-b')
    ax.plot(df_mls[nwater],df_mls['R'],'-k')    
    ax.plot([df[nwater].min(),df[nwater].max()],[0,0],'--k')
    ax.plot([csm[nwater],csm[nwater]],[df['R'].min(),df['R'].max()],'--k')
    ax.text(df_mls[nwater].iloc[-1],df_mls['R_energy'].iloc[-1],'$R_{\mathrm{energy}}$',
            color='r',ha='left',va='center')
    ax.text(df_mls[nwater][0],df_mls['R_water'][0],'$R_{\mathrm{water}}$',
            color='b',ha='center',va='bottom')
    ax.text(df_mls[nwater][0],df_mls['R'][0],'$R_{\mathrm{energy}}-R_{\mathrm{water}}$',
            color='k',ha='center',va='top')
    ax.text(csm[nwater],df['R'].min(),'$\ CSM=%.2f$' % (csm[nwater]),
            color='k',ha='left',va='bottom')
    ax.set_xlabel('Surface Soil Moisture [vol-%]')
    ax.set_ylabel('Correlation or Correlation Difference [1]')
    plt.savefig(os.path.join(path,"%s_None_dR.png" % (m.name)))
    plt.close()

    # Note: Plots and html pages are generated automatically from the
    # data in the netCDF files. In this case, we have made a custom
    # plot here. But we still need to add an entry that we will flag
    # as 'rendered'. This will get the figure added to the html page
    # in the right section.
    dR = Variable(da=xr.DataArray(data=df_mls['R'].values,coords={nwater:df_mls[nwater].values}),varname="dR")
    dR.setAttr("rendered",1)
    dR.setAttr("longname","Correlation Difference Distribution")
    
    # Build up outputs
    c_plot = {
        "dRmap" :  R,
        "dR"    : dR
    }
    _,c_plot = add_analysis_name(aname,{},c_plot)
    df = [['model',str(None),aname,'Critical Soil Moisture','scalar','1',float(csm.iloc[0][nwater])],
          ['model',str(None),aname,'Fraction Sub CSM'      ,'scalar','1',fless]]
    df = pd.DataFrame(df,columns=['Model','Region','Analysis','ScalarName','ScalarType','Units','Data'])
    return {},c_plot,df

class ConfLoCoMetrics(Confrontation):

    def __init__(self,**kwargs):
        
        self.source   = kwargs.get(  "source",None)
        self.variable = kwargs.get("variable","Loco Metrics")
        self.unit     = kwargs.get(    "unit",None)
        self.regions  = kwargs.get( "regions",[None])
        self.master   = kwargs.get(  "master",True)
        self.path     = kwargs.get(    "path","./")
        self.cmap     = kwargs.get(    "cmap",None)
        self.df_errs  = kwargs.get( "df_errs",None)
        self.df_plot  = None
        if not os.path.isdir(self.path): os.makedirs(self.path)

    def stageData(self,m,**kwargs):
        pass
    
    def confront(self,m,**kwargs):

        rplot = {}; cplot = {}; dfm = []

        # critical soil moisture
        rp,cp,df = critical_soil_moisture(m,path=self.path)
        rplot.update(rp); cplot.update(cp); dfm.append(df)
        
        # compute overall score and output
        dfm = pd.concat([df for df in dfm if len(df)>0],ignore_index=True)
        dfm.loc[dfm.Model=='model','Model'] = m.name
        dfm.to_csv(os.path.join(self.path,"%s.csv" % m.name),index=False)
        
        # output maps and curves
        ds = sanitize_into_dataset(cplot)
        ds.attrs = {'name':m.name,'color':m.color}
        ds.to_netcdf(os.path.join(self.path,"%s.nc" % m.name))
    
    def plotReference(self):
        pass
        
    def plotModel(self,m):

        if self.df_plot is None: self.df_plot = generate_plot_database(glob.glob(os.path.join(self.path,"*.nc")))
        df = self.df_plot[(self.df_plot.Model==m.name)]
        
        # Critical Soil Moisture Map
        dfs = df[df['Plot Name']=='dRmap']
        val = max(abs(dfs.iloc[0]['Plot Min']),abs(dfs.iloc[0]['Plot Max']))
        if len(dfs):
            dfs = dfs.iloc[0]
            v = Variable(filename=dfs['Filename'],varname=dfs['Variable'])
            v.plot(figsize=(6,4.5),tight_layout=True,vmin=-val,vmax=+val,
                   cmap="BrBG",cbar_kwargs={'label':'$\longleftarrow$ Water Limited      Energy Limited $\longrightarrow$',
                                            'pad':0.05,
                                            'orientation':'horizontal'})
            path = os.path.join(self.path,"%s_None_dRmap.png" % (m.name))
            plt.gca().set_title(m.name + " " + dfs['Longname'])
            plt.gcf().savefig(path)
            plt.close()
            
if __name__ == "__main__":
    from ModelResult import ModelResult
    import os

    ROOT = os.environ['ILAMB_ROOT']
    M = [
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/BCC-CSM2-MR"  ),name="BCC-CSM2-MR"  ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/CanESM5"      ),name="CanESM5"      ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/CESM2"        ),name="CESM2"        ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/GFDL-ESM4"    ),name="GFDL-ESM4"    ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/IPSL-CM6A-LR" ),name="IPSL-CM6A-LR" ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/MIROC-ES2L"   ),name="MIROC-ES2L"   ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/MPI-ESM1.2-HR"),name="MPI-ESM1.2-HR"),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/NorESM2-LM"   ),name="NormESM2-LM"  ),
        ModelResult(os.path.join(ROOT,"MODELS/CMIP6/UKESM1-0-LL"  ),name="UKESM1-0-LL"  ),
    ]
    print("Initialize models...")
    for m in M:
        m.findFiles()
        m.getGridInformation()
        print("  ",m.name)
    
    c = ConfLoCoMetrics(path = "./_loco")
    for m in M:
        try:
            c.confront(m)
            print(m.name,"success")
        except:
            print(m.name,"failed")
    for m in M:
        try:
            c.plotModel(m)
            print(m.name,"success")
        except:
            print(m.name,"failed")
        
    c.generateHTML()
