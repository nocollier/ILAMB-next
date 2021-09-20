"""
Critical Soil Moisture as given in:

https://github.com/pdirmeyer/l-a-cheat-sheets/blob/main/Coupling_metrics_V30_CriticalSM.pdf

"""
from ModelResult import ModelResult
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def CriticalSoilMoisture(m,**kwargs):
    """Computes CSM for a given model.

    t0
    tf
    veg
    water
    energy
    per0
    perf

    """
    
    # Parse confrontation options
    t0      = kwargs.get("t0","1980-01-01")
    tf      = kwargs.get("tf","2000-01-01")
    nveg    = kwargs.get("veg","evspsbl")
    nwater  = kwargs.get("water","mrsos")
    nenergy = kwargs.get("energy","ts")

    # Get the model variables and process
    veg    = m.getVariable(nveg   ,t0=t0,tf=tf)
    energy = m.getVariable(nenergy,t0=t0,tf=tf)
    water  = m.getVariable(nwater ,t0=t0,tf=tf)
    for v in [veg,energy,water]: v.detrend().decycle()

    # Note: In the paper, they compute correlations per season, but
    # their metric was for Europe and ours is global. The notion of
    # seasons does not transfer very well. Need to think up something
    # else as correlations not relevant when vegetation is not active.
    R_water  =  water.correlation(veg,dim='time')
    R_energy = energy.correlation(veg,dim='time')
    R = R_energy - R_water
    R.plot(figsize=(6,4.5),tight_layout=True,
           cmap="BrBG",cbar_kwargs={'label':'$\longleftarrow$ Water Limited      Energy Limited $\longrightarrow$',
                                    'pad':0.05,
                                    'orientation':'horizontal'})
    plt.savefig("dRmap_%s_%s_%s_%s.png" % (m.name,nveg,nenergy,nwater))
    plt.close()

    # Analysis includes looking at mopisture at transitions of
    # correlations. So we read the water variable again and attempt to
    # convert to a volumetric %
    water = m.getVariable(nwater,t0=t0,tf=tf).integrateInTime(mean=True)
    if nwater.lower() == "mrsos":
        density = 1000. # [kg m-3]
        thick   = 0.1   # [m]
        water.convert("kg m-2")
        water.ds[water.varname] /= density*thick
        water.ds[water.varname].attrs['units'] = "1"        
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

    # Scatter plot
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
    plt.savefig("dR_%s_%s_%s_%s.png" % (m.name,nveg,nenergy,nwater))


if __name__ == "__main__":

    m = ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2",name="CESM2")
    m.findFiles()
    m.getGridInformation()
    CriticalSoilMoisture(m)
