"""
Critical Soil Moisture as given in:

https://github.com/pdirmeyer/l-a-cheat-sheets/blob/main/Coupling_metrics_V30_CriticalSM.pdf

"""
from ModelResult import ModelResult
import matplotlib.pyplot as plt

m = ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2",name="CESM2")
m.findFiles()
m.getGridInformation()

t0 = "1980-01-01"
tf = "2000-01-01"
et    = m.getVariable("evspsbl",t0=t0,tf=tf)
ts    = m.getVariable("ts"     ,t0=t0,tf=tf)
mrsos = m.getVariable("mrsos"  ,t0=t0,tf=tf)
for v in [mrsos,et,ts]: v.detrend().decycle()

R_water  = mrsos.correlation(et,dim='time')
R_energy =    ts.correlation(et,dim='time')
R = R_energy - R_water
R.plot(cmap="BrBG")
plt.show()

