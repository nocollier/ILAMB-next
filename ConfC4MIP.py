from ILAMB.Confrontation import Confrontation,getVariableList
from ILAMB.Variable import Variable
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
        
        names = {'1pctCO2':'full','1pctCO2-bgc':'bgc','1pctCO2-rad':'rad','piControl':'ctl'}

        # check for the minimum required data
        if               not    m.children: raise ValueError("Model '%s' has no children"      % (m.name))
        if '1pctCO2'     not in m.children: raise ValueError("Model '%s' has no '1pctCO2'"     % (m.name))
        if '1pctCO2-bgc' not in m.children: raise ValueError("Model '%s' has no '1pctCO2-bgc'" % (m.name))
        
        # model nbp
        nbp = m.getVariable("nbp")
        y0,yf = np.round(nbp['1pctCO2'].time_bnds[[0,-1],[0,1]]/365)+1850
        years = np.asarray([np.arange(y0,yf),np.arange(y0+1,yf+1)]).T
        intervals = 365*(years-1850)
        nbp0 = {}
        for key in nbp.keys():
            nbp [key] = nbp[key].integrateInSpace().accumulateInTime().convert("Pg")
            nbp0[key] = nbp[key].data[0]
            nbp [key] = nbp[key].coarsenInTime(intervals)
            nbp[key].name = "nbp_%s" % names[key]

        # model fgco2
        fgco2  = m.getVariable("fgco2")
        fgco20 = {}
        for key in fgco2.keys():
            fgco2 [key] = fgco2[key].integrateInSpace().accumulateInTime().convert("Pg")
            fgco20[key] = fgco2[key].data[0]
            fgco2 [key] = fgco2[key].coarsenInTime(intervals)
            fgco2[key].name = "fgco2_%s" % names[key]

        # model tas
        tas  = m.getVariable("tas")
        tas0 = {}
        for key in tas.keys():
            tas [key] = tas[key].integrateInSpace(mean=True)
            tas0[key] = tas[key].data[0]
            tas [key] = tas[key].coarsenInTime(intervals)
            tas[key].name = "tas_%s" % names[key]
            
        # change in atmospheric carbon [ppm]
        dA = self.CO2_0*((1+self.rate)**(years[:,1]-y0)-1)
        
        # compute changes based on monthly data
        dL = {}; dO = {}; dT = {}
        for key in   nbp: dL[key] =   nbp[key].data - nbp0  [key]
        for key in fgco2: dO[key] = fgco2[key].data - fgco20[key]
        for key in   tas: dT[key] =   tas[key].data-  tas0  [key]

        # for reasons not clear to me, I am getting underflow errors
        # and so we use errstate
        with np.errstate(all='ignore'):

            # alpha/beta irrespective of presence of rad simulations
            alpha = dT['1pctCO2']/dA
            betaL = dL['1pctCO2-bgc']/dA
            betaO = dO['1pctCO2-bgc']/dA

            # gamma/gain based on the residual = full-bgc simulation
            gammaL = (dL['1pctCO2']-dL['1pctCO2-bgc'])/(dT['1pctCO2']-dT['1pctCO2-bgc'])
            gammaO = (dO['1pctCO2']-dO['1pctCO2-bgc'])/(dT['1pctCO2']-dT['1pctCO2-bgc'])            
            gain   = -alpha*(gammaL+gammaO)/(1+betaL+betaO)
        
            # gamma/gain based on the rad simulation
            gammaL_rad = gammaO_rad = gain_rad = None
            if ('1pctCO2-rad' in dL and '1pctCO2-rad' in dO and '1pctCO2-rad' in dT):
                gammaL_rad =  dL['1pctCO2-rad']/dT['1pctCO2-rad']
                gammaO_rad =  dO['1pctCO2-rad']/dT['1pctCO2-rad']
                gain_rad   = -alpha*(gammaL_rad+gammaO_rad)/(1+betaL+betaO)

        # beta / gamma unstable near beginning so we mask
        mask  = np.zeros(betaL.size,dtype=int); mask[:25] = 1
        betaL = np.ma.masked_array(betaL,mask=mask)
        betaO = np.ma.masked_array(betaO,mask=mask)
        gammaL = np.ma.masked_array(gammaL,mask=mask)
        gammaO = np.ma.masked_array(gammaO,mask=mask)
        gain = np.ma.masked_array(gain,mask=mask)
        if gammaL_rad is not None:
            gammaL_rad = np.ma.masked_array(gammaL_rad,mask=mask)
            gammaO_rad = np.ma.masked_array(gammaO_rad,mask=mask)
            gain_rad = np.ma.masked_array(gain_rad,mask=mask)
        
        with Dataset(os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name)),mode="w") as results:
            results.setncatts({"name":m.name,"color":m.color,"complete":0})
            t  = nbp['1pctCO2'].time
            tb = nbp['1pctCO2'].time_bnds

            # write out annual fluxes/states
            for key in   nbp:   nbp[key].toNetCDF4(results,group="Feedback")
            for key in fgco2: fgco2[key].toNetCDF4(results,group="Feedback")
            for key in   tas:   tas[key].toNetCDF4(results,group="Feedback")
            Variable(name="co2",unit="ppm",time=t,time_bnds=tb,data=dA+self.CO2_0).toNetCDF4(results,group="Feedback")
            Variable(name="betaL",unit="Pg ppm-1",time=t,time_bnds=tb,data=betaL).toNetCDF4(results,group="Feedback")
            Variable(name="betaO",unit="Pg ppm-1",time=t,time_bnds=tb,data=betaO).toNetCDF4(results,group="Feedback")
            Variable(name="gammaL",unit="Pg K-1",time=t,time_bnds=tb,data=gammaL).toNetCDF4(results,group="Feedback")
            Variable(name="gammaO",unit="Pg K-1",time=t,time_bnds=tb,data=gammaO).toNetCDF4(results,group="Feedback")
            Variable(name="gain",unit="Pg K-1",time=t,time_bnds=tb,data=gain).toNetCDF4(results,group="Feedback")
            if gammaL_rad is not None:
                Variable(name="gammaL (rad)",unit="Pg K-1",time=t,time_bnds=tb,data=gammaL_rad).toNetCDF4(results,group="Feedback")
                Variable(name="gammaO (rad)",unit="Pg K-1",time=t,time_bnds=tb,data=gammaO_rad).toNetCDF4(results,group="Feedback")
                Variable(name="gain (rad)",unit="Pg K-1",time=t,time_bnds=tb,data=gain_rad).toNetCDF4(results,group="Feedback")

            # write out scalars
            Variable(name="alpha",unit="K ppm-1" ,data=alpha[-1]).toNetCDF4(results,group="Feedback")
            Variable(name="betaL",unit="Pg ppm-1",data=betaL[-1]).toNetCDF4(results,group="Feedback")
            Variable(name="betaO",unit="Pg ppm-1",data=betaO[-1]).toNetCDF4(results,group="Feedback")
            Variable(name="gammaL (FC-BGC)",unit="Pg K-1",data=gammaL[-1]).toNetCDF4(results,group="Feedback")
            if gammaL_rad is not None: Variable(name="gammaL (RAD)",unit="Pg K-1",data=gammaL_rad[-1]).toNetCDF4(results,group="Feedback")
            Variable(name="gammaO (FC-BGC)",unit="Pg K-1",data=gammaO[-1]).toNetCDF4(results,group="Feedback")
            if gammaO_rad is not None: Variable(name="gammaO (RAD)",unit="Pg K-1",data=gammaO_rad[-1]).toNetCDF4(results,group="Feedback")
            Variable(name="gain (FC-BGC)",unit="1",data=gain[-1]).toNetCDF4(results,group="Feedback")
            if gain_rad is not None: Variable(name="gain (RAD)",unit="1",data=gain_rad[-1]).toNetCDF4(results,group="Feedback")
            results.setncattr("complete",1)

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
            co2 = Variable(filename=fname,variable_name="co2"     ,groupname="Feedback")
            tas = Variable(filename=fname,variable_name="tas_full",groupname="Feedback")
            for vname in ['nbp','fgco2','tas']:
                f1,a1 = plt.subplots(figsize=(6,5),tight_layout=True)
                for var,clr in zip(['full','bgc','rad'],['b','g','r']):
                    pname = "%s_%s" % (vname,var)
                    if pname not in vlst: continue
                    v = Variable(filename=fname,variable_name=pname,groupname="Feedback")
                    a1.plot(v.time/365+1850,v.data,'-',color=clr,label=var.replace("full","FC").upper())
                _formatPlot(a1,"[%s]" % v.unit,vname=vname)
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
                    v = Variable(filename=fname,variable_name=pname,groupname="Feedback")
                    lbl = _pname2lbl(pname)
                    a2.plot(co2.data,v.data,lt,color=clr,label=lbl)
                _formatPlot(a2,"[%s]" % v.unit,xl="[%s]" % co2.unit,vname=vname)
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
                    v = Variable(filename=fname,variable_name=pname,groupname="Feedback")
                    lbl = _pname2lbl(pname)
                    a2.plot(tas.data-tas.data[0],v.data,lt,color=clr,label=lbl)
                _formatPlot(a2,"[%s]" % v.unit,xl="[%s]" % tas.unit,vname=vname)
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
                    v = Variable(filename=fname,variable_name=pname,groupname="Feedback")
                    lbl = _pname2lbl(pname)
                    a1.plot(v.time/365+1850,v.data,lt,color=clr,label=lbl)
            _formatPlot(a1,"[%s]" % v.unit,vname=vname)
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
    m = ModelResult("/home/nate/data/ILAMB/MODELS/1pctCO2/CanESM5",name="CanESM5r")
    m.findFiles(group_attr="experiment_id") # <-- equivalent way of defining groups
    del m.children['1pctCO2-rad']
    m.getGridInformation()
    M.append(m)

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


