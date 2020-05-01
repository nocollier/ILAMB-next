from ModelResult import ModelResult
from ConfC4MIP import ConfC4MIP
import os

def E3SM():
    fc  = ModelResult("/global/cfs/cdirs/m3522/1pctco2_temp/processed/data/1pctco2"   ,name="1pctCO2"    ).findFiles()
    bgc = ModelResult("/global/cfs/cdirs/m3522/1pctco2_temp/processed/data/1pctco2bgc",name="1pctCO2-bgc").findFiles()
    rad = ModelResult("/global/cfs/cdirs/m3522/1pctco2_temp/processed/data/1pctco2rad",name="1pctCO2-rad").findFiles()
    ctl = ModelResult("/global/cfs/cdirs/m3522/1pctco2_temp/processed/data/1pctco2ctl",name="piControl"  ).findFiles()
    m   = ModelResult("/global/cfs/cdirs/m3522/1pctco2_temp/processed/data/",name="E3SM")
    m.addModel([fc,bgc,rad,ctl])
    m.getGridInformation()
    for sm in [fc,bgc,rad,ctl]:
        sm.variables['tas'] = [f for f in sm.variables['tas'] if "Lmon" not in f]
    return m

M = []
M.append(E3SM())

path = "./C4MIP"
if not os.path.isdir(path): os.mkdir(path)
c = ConfC4MIP(name = "CarbonCycleFeedback",output_path = path)
for m in M:
    if not os.path.isfile(os.path.join(c.output_path,"CarbonCycleFeedback_%s.nc" % m.name)):
        c.confront(m)
c.determinePlotLimits()
for m in M:
    if not os.path.isfile(os.path.join(c.output_path,"CarbonCycleFeedback_%s.nc" % m.name)): continue
    c.modelPlots(m)
c.generateHtml()


