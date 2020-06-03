from ILAMB.Confrontation import Confrontation,getVariableList
from Variable import ilamb_variable
import xarray as xr
import ILAMB.Post as post
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os,glob

matplotlib.rc('font', **{'size':14})

class ConfC4MIP(Confrontation):

    def __init__(self,**kwargs):

        self.name        = kwargs.get("name",None)
        self.output_path = kwargs.get("output_path","./")
        self.master      = True
        self.rate        = 0.01
        self.CO2_0       = 284.7 # [ppm]
        self.regions     = ['global']
        self.keywords    = kwargs
        
        pages = []
        pages.append(post.HtmlPage("Feedback","By Model"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Global states and fluxes","Sensitivity parameters"])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlAllModelsPage("AllModels","All Models"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)

        self.layout = post.HtmlLayout(pages,self.name)
        
    def stageData(self,m):
        pass

    def confront(self,m):
        
        names   = {'1pctCO2':'full','1pctCO2-bgc':'bgc','1pctCO2-rad':'rad','piControl':'ctl'}
        data    = {}
        scalars = {}

        # check for the minimum required data
        if               not    m.children: raise ValueError("Model '%s' has no children"      % (m.name))
        if '1pctCO2'     not in m.children: raise ValueError("Model '%s' has no '1pctCO2'"     % (m.name))
        if '1pctCO2-bgc' not in m.children: raise ValueError("Model '%s' has no '1pctCO2-bgc'" % (m.name))
        
        # process nbp
        nbp = m.getVariable("nbp")
        for key in nbp.keys():
            nbp[key] = nbp[key].ilamb.integrateInSpace()
            nbp[key] = nbp[key].ilamb.accumulateInTime()
            nbp[key] = nbp[key].ilamb.convert("Pg")
            nbp[key].name = "nbp_%s" % names[key]
            
        # process fgco2
        fgco2 = m.getVariable("fgco2")
        for key in fgco2.keys():
            fgco2[key] = fgco2[key].ilamb.integrateInSpace()
            fgco2[key] = fgco2[key].ilamb.accumulateInTime()
            fgco2[key] = fgco2[key].ilamb.convert("Pg")
            fgco2[key].name = "fgco2_%s" % names[key]

        # process tas
        tas = m.getVariable("tas")
        for key in tas.keys():
            tas[key] = tas[key].ilamb.integrateInSpace(mean=True)
            tas[key] = tas[key].ilamb.convert("degC")
            tas[key].name = "tas_%s" % names[key]
       
        # compute changes based on monthly data, then coarsen to annual
        dL = {}; dO = {}; dT = {}
        for key in nbp:
            dL[key] = (nbp[key]-nbp[key][0]).coarsen(time=12,boundary='trim').mean()
            dL[key].attrs['units'] = nbp[key].attrs['units']
            data['nbp_%s' % names[key]] = nbp[key].coarsen(time=12,boundary='trim').mean()
        for key in fgco2:
            dO[key] = (fgco2[key]-fgco2[key][0]).coarsen(time=12,boundary='trim').mean()
            dO[key].attrs['units'] = fgco2[key].attrs['units']
            data['fgco2_%s' % names[key]] = fgco2[key].coarsen(time=12,boundary='trim').mean()
        for key in tas:
            dT[key] = (tas[key]-tas[key][0]).coarsen(time=12,boundary='trim').mean()
            dT[key].attrs['units'] = tas[key].attrs['units']
            data['tas_%s' % names[key]] = tas[key].coarsen(time=12,boundary='trim').mean()

        # change in atmospheric carbon [ppm]
        t = np.array([t.total_seconds()/(3600*24*365) for t in dT['1pctCO2'].time.data-dT['1pctCO2'].time.data[0]])+1
        dA = xr.DataArray(self.CO2_0*((1+self.rate)**t-1),
                          dims=('time'),
                          coords={'time':dT['1pctCO2'].time},
                          attrs={'units':'ppm'},
                          name="co2")
        data[dA.name] = dA
            
        # alpha/beta irrespective of presence of rad simulations
        alpha = dT['1pctCO2']/dA
        betaL = dL['1pctCO2-bgc']/dA
        betaO = dO['1pctCO2-bgc']/dA
        
        # gamma/gain based on the residual = full-bgc simulation
        gammaL = (dL['1pctCO2']-dL['1pctCO2-bgc'])/(dT['1pctCO2']-dT['1pctCO2-bgc'])
        gammaO = (dO['1pctCO2']-dO['1pctCO2-bgc'])/(dT['1pctCO2']-dT['1pctCO2-bgc'])            
        gain   = -alpha*(gammaL+gammaO)/(1+betaL+betaO)

        # beginning of gamma/gain is unstable, skip
        skip = 20
        gammaL[:skip] = np.nan
        gammaO[:skip] = np.nan
        gain  [:skip] = np.nan
        
        data.update({"alpha":alpha,"betaL":betaL,"betaO":betaO,"gammaL":gammaL,"gammaO":gammaO,"gain":gain})
        scalars.update({"alpha":xr.DataArray(alpha.data[-1],attrs={"units":"K ppm-1"}),
                        "betaL":xr.DataArray(betaL.data[-1],attrs={"units":"Pg ppm-1"}),
                        "betaO":xr.DataArray(betaO.data[-1],attrs={"units":"Pg ppm-1"}),
                        "gammaL (FC-BGC)":xr.DataArray(gammaL.data[-1],attrs={"units":"Pg K-1"}),
                        "gammaO (FC-BGC)":xr.DataArray(gammaO.data[-1],attrs={"units":"Pg K-1"}),
                        "gain (FC-BGC)":xr.DataArray(gain.data[-1],attrs={"units":"1"})})
        
        # division of xarrays does not account for units changes
        alpha.attrs['units'] = "%s / (%s)" % (dT['1pctCO2'].attrs['units'],dA.attrs['units'])
        betaL.attrs['units'] = "%s / (%s)" % (dL['1pctCO2'].attrs['units'],dA.attrs['units'])
        betaO.attrs['units'] = "%s / (%s)" % (dO['1pctCO2'].attrs['units'],dA.attrs['units'])
        gammaL.attrs['units'] = "%s / (%s)" % (dL['1pctCO2'].attrs['units'],dT['1pctCO2'].attrs['units'])
        gammaO.attrs['units'] = "%s / (%s)" % (dO['1pctCO2'].attrs['units'],dT['1pctCO2'].attrs['units'])
        gain.attrs['units'] = "1"
        
        # gamma/gain based on the rad simulation
        gammaL_rad = gammaO_rad = gain_rad = None
        if ('1pctCO2-rad' in dL and '1pctCO2-rad' in dO and '1pctCO2-rad' in dT):
            gammaL_rad =  dL['1pctCO2-rad']/dT['1pctCO2-rad']
            gammaO_rad =  dO['1pctCO2-rad']/dT['1pctCO2-rad']
            gain_rad   = -alpha*(gammaL_rad+gammaO_rad)/(1+betaL+betaO)
            data.update({"gammaL (rad)":gammaL_rad,"gammaO (rad)":gammaO_rad,"gain (rad)":gain_rad})
            scalars.update({"gammaL (RAD)":xr.DataArray(gammaL_rad.data[-1],attrs={"units":"Pg K-1"}),
                            "gammaO (RAD)":xr.DataArray(gammaO_rad.data[-1],attrs={"units":"Pg K-1"}),
                            "gain (RAD)":xr.DataArray(gain_rad.data[-1],attrs={"units":"1"})})
            gammaL_rad.attrs['units'] = "%s / (%s)" % (dL['1pctCO2'].attrs['units'],dT['1pctCO2'].attrs['units'])
            gammaO_rad.attrs['units'] = "%s / (%s)" % (dO['1pctCO2'].attrs['units'],dT['1pctCO2'].attrs['units'])
            gain_rad.attrs['units'] = "1"

            # beginning of gamma/gain is unstable, skip
            gammaL_rad[:skip] = np.nan
            gammaO_rad[:skip] = np.nan
            gain_rad  [:skip] = np.nan
            
        # write out data
        fname = os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name))
        with Dataset(fname,mode="w") as dset: dset.setncatts({"name":m.name,"color":m.color,"complete":0})
        xr.Dataset(   data).to_netcdf(path=fname,group="Feedback"        ,mode="a")
        xr.Dataset(scalars).to_netcdf(path=fname,group="Feedback/scalars",mode="a")
        with Dataset(fname,mode="r+") as dset: dset.setncatts({"complete":1})
        
    def determinePlotLimits(self):

        def _update(thing,data,limits):
            limits[thing]["min"] = min(limits[thing]["min"],d.min())
            limits[thing]["max"] = max(limits[thing]["max"],d.max())
            return limits
        
        # initialize limits
        limits = {}
        for thing in ['carbon','tas','beta','gamma','gain']:
            limits[thing] = {"min":1e20,"max":-1e20}

        # find limits
        for fname in glob.glob(os.path.join(self.output_path,"*.nc")):
            with Dataset(fname) as dset:
                grp  = dset.groups["Feedback"]
                vlst = getVariableList(grp)
                for v in vlst:
                    d = grp.variables[v][...]
                    if "nbp" in v or "fgco2" in v: limits = _update("carbon",d,limits)
                    for thing in ["tas","beta","gamma","gain"]:
                        if thing in v: limits = _update(thing,d,limits)

        # add buffer
        for thing in ['carbon','tas','beta','gamma','gain']:
            dL = 0.05*(limits[thing]['max']-limits[thing]['min'])
            limits[thing]["min"] -= dL
            limits[thing]["max"] += dL
            
        self.limits = limits
        
    def modelPlots(self,m):
        def _formatPlot(ax,yl,xl=None,vname=None):
            ax.set_title("")
            if xl: ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            h,l = ax.get_legend_handles_labels()
            nc = len(h)
            if nc > 3: nc = 2
            ax.legend(bbox_to_anchor=(0,1.005,1,0.25),loc='lower left',mode='expand',ncol=nc,borderaxespad=0,frameon=False)
            ax.grid('on')
            if vname is None: return
            if vname in ['nbp','fgco2']: ax.set_ylim(self.limits["carbon"]["min"],self.limits["carbon"]["max"])
            if vname in ['tas','beta','gamma','gain']: ax.set_ylim(self.limits[vname]["min"],self.limits[vname]["max"])
        def _pname2lbl(n):
            l = n.replace("betaL","$\\beta_{L}$")
            l = l.replace("betaO","$\\beta_{O}$")
            l = l.replace("gammaL","$\\gamma_{L}$")
            l = l.replace("gammaO","$\\gamma_{O}$")
            if "gamma" in n or "gain" in n:
                l = l.replace("(rad)","(RAD)")
                if "RAD" not in l: l += " (FC-BGC)"
            return l
        header = {}
        header["nbp"]   = "Land Accumulated Carbon"
        header["fgco2"] = "Ocean Accumulated Carbon"
        header["tas"]   = "Temperature"
        header["beta"]  = "Carbon Sensitivity to CO2"
        header["gamma"] = "Carbon Sensitivity to Climate Change"
        header["gain"]  = "Climate-Carbon Feedback Gain"
        fname = os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name))
        if not os.path.isfile(fname): return
        page = [page for page in self.layout.pages if "Feedback" in page.name][0]
        with Dataset(fname) as dset:
            grp  = dset.groups["Feedback"]
            vlst = getVariableList(grp)
            co2 = xr.open_dataset(fname,group="Feedback")["co2"]
            tas = xr.open_dataset(fname,group="Feedback")["tas_full"]
            for vname in ['nbp','fgco2','tas']:
                f1,a1 = plt.subplots(figsize=(6,5),tight_layout=True)
                for var,clr in zip(['full','bgc','rad'],['b','g','r']):
                    pname = "%s_%s" % (vname,var)
                    if pname not in vlst: continue
                    v = xr.open_dataset(fname,group="Feedback")[pname]
                    v.plot(ax=a1,color=clr,label=var.replace("full","FC").upper())
                _formatPlot(a1,"[%s]" % v.units,vname=vname)
                f1.savefig(os.path.join(self.output_path,"%s_global_%s.png" % (m.name,vname)))
                page.addFigure("Global states and fluxes",
                               "%s" % vname,
                               "MNAME_global_%s.png" % vname,
                               side   = header[vname],
                               longname = header[vname],
                               legend = False)
                plt.close()
                
            vname = "beta"
            f2,a2 = plt.subplots(figsize=(6,5),tight_layout=True)
            for suf,lt in zip(['',' (rad)'],['-','--']):
                for var,clr in zip(['L','O'],
                                   [np.asarray([95,184,104])/255,np.asarray([51,175,255])/255]):
                    pname = "%s%s%s" % (vname,var,suf)
                    if pname not in vlst: continue
                    v = xr.open_dataset(fname,group="Feedback")[pname]
                    lbl = _pname2lbl(pname)
                    a2.plot(co2-co2[0],v,lt,color=clr,label=lbl)
                _formatPlot(a2,"[%s]" % v.units,xl="[%s]" % co2.units,vname=vname)
                f2.savefig(os.path.join(self.output_path,"%s_global_%s.png" % (m.name,vname)))
                page.addFigure("Sensitivity parameters",
                               "%s" % vname,
                               "MNAME_global_%s.png" % vname,
                               side   = header[vname],
                               longname = header[vname],
                               legend = False)
                plt.close()
                
            vname = "gamma"
            f2,a2 = plt.subplots(figsize=(6,5),tight_layout=True)
            for suf,lt in zip(['',' (rad)'],['-','--']):
                for var,clr in zip(['L','O'],
                                   [np.asarray([95,184,104])/255,np.asarray([51,175,255])/255]):
                    pname = "%s%s%s" % (vname,var,suf)
                    if pname not in vlst: continue                    
                    v = xr.open_dataset(fname,group="Feedback")[pname]
                    lbl = _pname2lbl(pname)
                    a2.plot(tas-tas[0],v,lt,color=clr,label=lbl)
                _formatPlot(a2,"[%s]" % v.attrs['units'],xl="[%s]" % tas.attrs['units'],vname=vname)
                f2.savefig(os.path.join(self.output_path,"%s_global_%s.png" % (m.name,vname)))
                page.addFigure("Sensitivity parameters",
                               "%s" % vname,
                               "MNAME_global_%s.png" % vname,
                               side   = header[vname],
                               longname = header[vname],
                               legend = False)
                plt.close()
                
            vname = "gain"
            f1,a1 = plt.subplots(figsize=(6,5),tight_layout=True)
            for suf,lt in zip(['',' (rad)'],['-','--']):
                for var,clr in zip(['',],['k']):
                    pname = "%s%s%s" % (vname,var,suf)
                    if pname not in vlst: continue                    
                    v = xr.open_dataset(fname,group="Feedback")[pname]
                    lbl = _pname2lbl(pname)
                    v.plot(ax=a1,linestyle=lt,color=clr,label=lbl)
            _formatPlot(a1,"[%s]" % v.attrs['units'],vname=vname)
            f1.savefig(os.path.join(self.output_path,"%s_global_%s.png" % (m.name,vname)))
            page.addFigure("Sensitivity parameters",
                           "%s" % vname,
                           "MNAME_global_%s.png" % vname,
                           side   = header[vname],
                           longname = header[vname],
                           legend = False)
            plt.close()
            
if __name__ == "__main__":

    # initialize the model
    from ModelResult import ModelResult
    M = []

    # first model has all 3
    m = ModelResult("/home/nate/data/ILAMB/MODELS/1pctCO2/CanESM5",name="CanESM5")
    m.findFiles(group_regex=".*_(.*)_r1i1p1f1*")
    m.getGridInformation()
    M.append(m)

    # second model we will remove the rad runs
    #m = ModelResult("/home/nate/data/ILAMB/MODELS/1pctCO2/CanESM5",name="CanESM5r")
    #m.findFiles(group_attr="experiment_id") # <-- equivalent way of defining groups
    #del m.children['1pctCO2-rad']
    #m.getGridInformation()
    #M.append(m)

    # initialize the confrontation
    path = "./C4MIP"
    if not os.path.isdir(path): os.mkdir(path)
    c = ConfC4MIP(name = "CarbonCycleFeedback",output_path = path)
    for m in M:
        if not os.path.isfile(os.path.join(c.output_path,"CarbonCycleFeedback_%s.nc" % m.name)):
            c.confront(m)
    c.determinePlotLimits()
    for m in M: c.modelPlots(m)
    c.generateHtml()


