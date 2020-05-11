from ModelResult import ModelResult
from ConfC4MIP import ConfC4MIP
import pandas as pd
import os,re
    
def SetupModels():
    df = pd.read_csv('../CMIP6Database/cmip6_stores.csv')
    V = ['tas','fgco2','nbp','areacella','areacello','sftlf']
    E = ['1pctCO2','1pctCO2-bgc']
    T = ['Lmon','Amon','Emon','Omon','fx','Ofx']
    query = []
    query.append("( %s )" % (" | ".join(["experiment_id == '%s'" % e for e in E])))
    query.append("( %s )" % (" | ".join(["variable_id == '%s'" % v for v in V])))
    query.append("( %s )" % (" | ".join(["table_id == '%s'" % t for t in T])))
    query = " & ".join(query)
    df_q = df.query(query)
    models = df_q.source_id.unique()
    accept = []
    for m in models:
        df_m = df_q.query("source_id == '%s'" % (m))
        experiments = df_m.experiment_id.unique()
        variables   = df_m.variable_id.unique()
        members     = df_m.member_id.unique()
        grids       = df_m.grid_label.unique()
        if not set(experiments)==set(E): continue
        if not set(variables)  ==set(V): continue
        for member in members:
            for grid in grids:
                ok = True
                for exp in experiments:
                    if set(variables)!=set(V): ok = False
                if ok: accept.append((m,member,grid))
    def _convert(x):
        m = re.search("r(\d+)i(\d+)p(\d+)f(\d+)",x)
        if m:
            return tuple([int(e) for e in m.groups()])
        return (0,0,0,0)
    accept = sorted(accept,key=lambda x:(x[0].lower(),) + _convert(x[1]) + (x[2],))
    pairs = [accept[0]]
    for a in accept:
        if a[0] != pairs[-1][0]: pairs.append(a)
    M = []
    for a in pairs:
        df_m = df_q.query("( source_id == '%s' & member_id == '%s' & grid_label == '%s' )" % a)
        paths = list(df_m.path)
        for i,p in enumerate(paths): paths[i] = os.path.join(os.environ["CMIP6_DIR"],p)
        m = ModelResult(None,name=a[0])
        m.findFiles(file_paths=paths,group_regex=".*_(.*)_%s*" % a[1])
        m.getGridInformation()
        M.append(m)
    return M

hostname = os.uname()[1]

def CanESM():
    m = ModelResult("/home/nate/data/ILAMB/MODELS/1pctCO2/CanESM5",name="CanESM5")
    m.findFiles(group_regex=".*_(.*)_r1i1p1f1*")
    m.getGridInformation()
    return m
    
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

M = SetupModels()

"""
path = "./C4MIP"
if not os.path.isdir(path): os.mkdir(path)
c = ConfC4MIP(name = "CarbonCycleFeedback",output_path = path)
for m in M:
    if not os.path.isfile(os.path.join(c.output_path,"CarbonCycleFeedback_%s.nc" % m.name)):
        try:
            print("confronting %s..." % m.name)
            c.confront(m)
            print("pass")
        except:
            print("fail")
            
c.determinePlotLimits()
for m in M:
    if not os.path.isfile(os.path.join(c.output_path,"CarbonCycleFeedback_%s.nc" % m.name)): continue
    c.modelPlots(m)
c.generateHtml()


"""
