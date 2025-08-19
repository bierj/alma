import matplotlib.pylab as plt
import matplotlib
import numpy as np
import pickle
from imtools import sigClip
from vis_sample.file_handling import import_data_ms
from glob import glob

matplotlib.rcParams.update({'font.size': 16})

c=2.99792e8

viss = sorted(glob('*final.timebin.flagged'))

def getBinnedVis(obsVis,binWid,binOffset=0,uvmax=None):
    data_vis=import_data_ms(obsVis)
    f0 = 0.5*(data_vis.freqs[-1]+data_vis.freqs[0])[0]
    re=np.squeeze(np.real(data_vis.VV))
    amp = np.squeeze(np.abs(data_vis.VV))
    im = np.squeeze(np.imag(data_vis.VV))
    visi = np.squeeze(data_vis.VV)
    uv = (data_vis.vv**2+data_vis.uu**2)**0.5 * c/f0
    visRMS = np.nanstd(np.real(data_vis.VV))

    if uvmax != None:
        uvInds = np.where(uv <= uvmax)
        uv = uv[uvInds]
        re = re[uvInds]
        amp = amp[uvInds]
        im = im[uvInds]
        visi = visi[uvInds]

    bins = np.arange(binOffset,uv.max(),binWid)
    binInds = np.digitize(uv,bins)

    sorter=np.argsort(uv)
    uvsorted=uv[sorter]
    rms = np.zeros(len(uv)) + visRMS

    uvBinned = []
    yerrs = []
    for i in range(len(bins)):
        if (np.sum(binInds==i) > 0):
            uvBinned.append(np.average(uv[binInds==i]))
            yerrs.append(1.29/np.sum(data_vis.VV.shape[-1]/rms[binInds==i]**2)**0.5)

    reBinned = []
    ampBinned = []
    imBinned = []
    for i in range(len(bins)):
        if (np.sum(binInds==i) > 0):
            vb = np.average(visi[:,:][binInds==i])
            reBinned.append(np.real(vb))
            ampBinned.append(np.abs(vb))
            imBinned.append(np.imag(vb))
            #reBinned.append(np.average(re[:,:][binInds==i]))
            #ampBinned.append(np.average(amp[:,:][binInds==i]))
            #imBinned.append(np.average(im[:,:][binInds==i]))

    return uvBinned, ampBinned, reBinned, imBinned, yerrs


uvBinneds = []
ampBinneds = []
yerrz = []
reBinneds = []

for vis in viss:
    uvBinned, ampBinned, reBinned, imBinned, yerrs = getBinnedVis(vis,binWid=12,binOffset=0,uvmax = 200)
    uvBinneds.append(uvBinned)
    ampBinneds.append(ampBinned)
    reBinneds.append(reBinned)
    yerrz.append(yerrs)

fig,axes = plt.subplots(1,1,figsize=(15,10))

for i in range(len(viss)):
    axes[0].errorbar(uvBinneds[i],reBinneds[i],yerrz[i],marker='o',linestyle=' ',markersize=3,label=viss[i])

axes[0].set_ylabel('Real (Jy)')
axes[0].set_xlabel('uv-distance (m)')


plt.tight_layout()

plt.savefig('myVis.pdf')

plt.show()
