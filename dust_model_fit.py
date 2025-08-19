#! /usr/bin/env python3

#################################################################
# Levenberg-Marquardt fitting routine to compare point-source,  #
# disk, or point-source plus 1/sigma (dust coma) models         #  
# in the Fourier plane                                          #
#################################################################

import os
import pickle
import sys

import casatasks
import casatools
import matplotlib.pyplot as plt
import numpy as np
import vis_sample as vs
from imtools import sigClip
from lmfit import Parameters, minimize
from scipy import special
#from pymodules.mpfit import mpfit
from scipy.interpolate import interp1d
from scipy.stats import chi2
from vis_sample.file_handling import import_data_ms

tb = casatools.table()

chisqtol = 0.01
xtol = 1.0e-4

c = 2.99792458e8
###################################################################
#Function to interface with LMFIT and return normalized residuals #
###################################################################

def getChisq(p):

    global model

    getChisq.nCalls += 1

    p.pretty_print(oneline=False, precision=5, fmt='e', columns=['value','vary'])

    #Generate the model
    #For a point-source, V(sigma) = pointFlux

    #For a 1/rho coma, V(sigma) = K * c / (sigma * nu) = K * lambda / sigma

    #pointFlux, comaFlux = p
    
    model = np.asarray([( p['pointFlux'] + (c / restFreq) * p['comaFlux']  / r)  for r in uvb_concat])


    residuals = np.ravel((reb_concat - np.real(model)) / yerr_concat[:,np.newaxis])

    print('Chisq ({:d} = '.format(getChisq.nCalls), np.sum(residuals**2),"\n")
    return (residuals)

#######################################
# Function to import the visibilities #
#######################################
def import_my_vis(vis,index=1.93,freq=459e9):
    obsVis = import_data_ms(vis)

    f = 0.5*(obsVis.freqs[-1] + obsVis.freqs[0])[0]
    print(f'{f:.4e}')
    freqs = np.squeeze(obsVis.freqs)
    uv = (obsVis.vv**2 + obsVis.uu**2)**0.5 * c / f
    re = np.squeeze(np.real(obsVis.VV))
    im = np.squeeze(np.imag(obsVis.VV))
    wgts = obsVis.wgts
    re_scale = np.zeros((obsVis.VV.shape[0],obsVis.VV.shape[1]))
    im_scale = np.zeros((obsVis.VV.shape[0],obsVis.VV.shape[1]))
    for i in range(len(freqs)):
        re_scale[:,i] = re[:,i] * (freq/freqs[i])**index
        im_scale[:,i] = im[:,i] * (freq/freqs[i])**index

    #Discard bad 0th and 1th rows
    re_scale = re_scale[:,2:]
    im_scale = im_scale[:,2:]
    return uv, re_scale, im_scale, wgts

def getBinnedVis(uv,re,binWid,binOffset,uvmax=None,wgts=None):
    visRMS = np.nanstd(re)

    if wgts is not None:
        rms = 1 / np.sqrt(wgts)
    else:
        rms = np.zeros(len(uv)) + visRMS

    if uvmax != None:
        uvInds = np.where(uv <= uvmax)
        uv = uv[uvInds]
        re = re[uvInds]
        rms = rms[uvInds]


    bins = np.arange(binOffset,uv.max(),binWid)
    binInds = np.digitize(uv,bins)

    uvBinned = []
    yerrs = []
    for i in range(len(bins)):
        if (np.sum(binInds==i) > 0):
            uvBinned.append(np.average(uv[binInds==i]))
            yerrs.append(1.29/np.sum(re.shape[-1]/rms[binInds==i]**2)**0.5)

    reBinned = []
    for i in range(len(bins)):
        if (np.sum(binInds==i) > 0):
            reBinned.append(np.average(re[:,:][binInds==i]))

    return np.asarray(uvBinned), np.asarray(reBinned), np.asarray(yerrs)


#Load the input parameters
exec(compile(open(sys.argv[1], "rb").read(), sys.argv[1], 'exec'))

#Read in the data
uv1, re_scale1, im_scale1, wgts1 = import_my_vis(visFile1)
uv2, re_scale2, im_scale2, wgts2 = import_my_vis(visFile2)

uv_concat = np.concatenate([uv1,uv2])
re_concat = np.concatenate([re_scale1,re_scale2])
wgt_concat = np.concatenate([wgts1,wgts2])


uvb_concat, reb_concat, yerr_concat = getBinnedVis(uv_concat, re_concat, binWid, binOffset, uvmax)



# ruv = np.sqrt(obsVis.uu**2 + obsVis.vv**2) * c / restFreq

#Set tolerance
ftol = chisqtol / len(reb_concat)



# #Set up the parameters
params = Parameters()
parnames = ['pointFlux','comaFlux']

i=0
for parname in parnames:
    params.add(parname, value=p0[i], vary = (not bool(pfix[i])))
    params[parname].min = 0.
    i+=1



getChisq.nCalls = 0
#If some parameters are free, do the fitting
if sum(pfix) != len(pfix):
    print("Starting LMFIT...")
    result = minimize(getChisq, params, xtol=xtol, ftol=ftol, epsfcn=1e-5, scale_covar=False)

    pf = result.params

    #Final run with optimized parameters to generate best-fit file and plots
    print('\nFinal call to function...')
    getChisq(pf)

    print('*******************************************')
    print('LMFIT exit: ', result.message)
    print('*******************************************')
   
    print('Number of calls to getChisq: ', getChisq.nCalls)
   
    # Reduced chi square
    print('\nReduced Chisq X_r = %.5f'% (result.redchi))
    print('P (probability that model is different from data due to chance) = %.3f\n' % chi2.sf(result.chisqr,result.nfree))

    pfsigmas = {}
    for parname in parnames:
        try:
            pfsigmas[parname] = pf[parname].stderr + 0.
        except:
            pfsigmas[parname] = 0.

    print('\nBest-fit parameters and 1-sigma covariance errors')
    print('Point-Flux = {:8.4e} +- {:8.4e}'.format(pf['pointFlux'].value,pfsigmas['pointFlux']))
    print('Power-Flux = {:8.4e} +- {:8.4e}'.format(pf['comaFlux'].value,pfsigmas['comaFlux']))
    print("Model run parameters written to "+outPars)

    #Write to the output file
    f = open(outPars, 'w')
    f.write(open(sys.argv[1], "r").read())
    f.write("\n\n******************************\n")
    f.write("# LMFIT Results: "+result.message)
    f.write('\n********************************\n')
    f.write('\nChi-Squre = {:.8f}\n'.format(result.chisqr))
    f.write('\n Reduced Chisq X_r = {:.8f}\n'.format(result.redchi))
    f.write('\nBest-fit parameters and 1-sigma covariance errors:\n')
    f.write('Point-Flux = {:8.4e} +- {:8.4e}\n'.format(pf['pointFlux'].value,pfsigmas['pointFlux']))
    f.write('Power-Flux = {:8.4e} +- {:8.4e}\n'.format(pf['comaFlux'].value,pfsigmas['comaFlux']))
    f.close()

    #Write the the model to an output file
    finalModel = np.vstack((uvb_concat,model)).T
    np.savetxt(outModel,finalModel)

#If all fixed, calculate model anyways
else:
    print("All parameters fixed")
    resids = getChisq(params)
    chisq = np.sum(resids**2)

#Plot model fit

fig = plt.figure()
plt.errorbar(uvb_concat,reb_concat,yerr_concat,marker='o',linestyle=' ',color='k')
plt.plot(uvb_concat,np.real(model),marker='s',linestyle=' ',color='r')
plt.show()

