'''
Mix wfs anf hits from different pyreco outputs into one file, shuffling but preserving hit and label pairs
Takes even ammount of arguments
format is:
    python mixer.py wfs1 wfs2 ... wfsn hits1 hits2 ... hitsn outprefix
    IMPORTANT for this script to work, the wfsn and hitsn format must be the following, otherwise it could lead to incorrect pairings!
        xxma_xov_wfs.txt and xxma_xov_hits.txt 
    No other parameter implemented yet

'''

import numpy as np
import csv
import sys 

args = sys.argv[1:-1]
n_pars = 2                          #change manually
p = sys.argv[-1]


assert len(args) % 2==0, "Provide even number of hit-wfs pairs" 
arg = []
for i in range(int(len(args)/2)):
    arg.append([args[i],args[i+int(len(args)/2)]])
args = np.asarray(arg,dtype=object).reshape(int(len(args)/2),2)

#create storage arrays, first element will be deleted later
ws = np.empty((1,4000))
hs = np.empty((1,),dtype=str)
ps = np.empty((1,n_pars),dtype=float)                          # this is the array containing vov, gate, and other information

for i in range(len(args)):
    print("Working on pair ", i, "/",len(args)-1)
    f = args[i,0]
    ma,ov = float(f[:f.find("ma_")]), float(f[f.find("ov_")-1:f.find("ov_")])
    w = np.loadtxt(args[i,1])
    n_ws  = len(w)
    ws = np.concatenate((ws,w))
    hs = np.concatenate((hs,np.genfromtxt(args[i,0],dtype=str)),dtype=str)
    ps = np.concatenate((ps,np.tile([ma,ov],n_ws).reshape(n_ws,n_pars)))

#get rid of first empty element
ws = ws[1:]
hs = hs[1:]    
ps = ps[1:]

#mix the outputs
idxs = np.random.permutation(len(ws))
ws   = ws[idxs]
hs   = hs[idxs]
ps   = ps[idxs]

#save the outputs
print("saving output...")
np.savetxt(p+"_hits.txt",hs,fmt='%s')
np.savetxt(p+"_wfs.txt",ws)
np.savetxt(p+"_pars.txt",ps)

