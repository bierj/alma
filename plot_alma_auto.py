import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from matplotlib.patches import Arc
from matplotlib.patches import FancyArrowPatch
from imtools import sigClip

def plot_alma(contfile, clevel, arrows, psa, ps_amv, ifrac, delta, figname, figtitle, prange):
    """contfile is continuum FITS file
    clevel is contour level - 3-sigma increments, 5-sigma increments, etc
    arrows - 1 includes sun/dust arrows, 0 turns off
    psa, ps_amv - retrieved from Horizons, check quantity 27, read description of these items
    ifrac - illuminated fraction,from Horizons, check quantity 10
    delta - geocentric distance (au)
    figname - name of the output figure
    figtitle - plot title for the figure
    prange - range of the plot (km), will plot from -x km to +x km"""

    #Set some plotting options
    # matplotlib.use('TkAgg')
    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    #plt.style.use('dark_background')
    

    #Define an astronomical unit in km
    AU = 1.49598e8

    #Read in continuum file
    hcont = fits.open(contfile)
    cont = hcont[0]

    #Get beam properties
    hdr = hcont[0].header
    bmaj = delta*AU*np.tan(hdr['BMAJ']*np.pi/180.)
    bmin = delta*AU*np.tan(hdr['BMIN']*np.pi/180.)
    bpa = hdr['BPA']

    #Get spatial axis
    str1 = hdr['CDELT2']
    pscl = float(str1)*3600.
    xscl = delta*AU*np.tan((pscl/3600.)*np.pi/180.)
    nside = int(prange/xscl)
    npix = 2*nside + 1
    xcnt = (np.arange(0, npix, dtype=float) - nside) * xscl

    #Extract continuum signal in mJy
    cdata = cont.data[:,:]*1000.
    #Find the peak pixel (center)
    c1max = np.where(cdata == np.nanmax(cdata))
    pcen = [c1max[1][0],c1max[0][0]]

    #Do some sigma clipping and measure the RMS
    contSignal = cdata
    rms1 = np.nanstd(contSignal)
    masked_cont = sigClip(contSignal,2)
    contrms = np.nanstd(masked_cont)
    cmax_val = cdata[c1max]

    #Extract the signal within the specified nucleocentric distance range
    contSignal = cdata[pcen[1]-nside:pcen[1]+nside,pcen[0]-nside:pcen[0]+nside]
    print('contrms: {}'.format(contrms))
    
    #Set some labels and the color map rangex
    rms = contrms
    zmin = np.nanmin(contSignal)
    zmax = np.nanmax(contSignal)
    barlabel = 'Continuum Flux (mJy beam$^{-1}$)'
    slabel = '$\sigma$ = %.2f mJy beam$^{-1}$'%(rms)


    #Define various options for contourlevels
    if clevel == 2:
        clevels = [-8,-6,-4,-2,2,4,6,8]
        contourlevels = [x*rms for x in clevels]
        strs = [str(i)+'$\sigma$' for i in clevels]
    if clevel == 1:
        clevels = [-5,-4,-3,-2,2,3,4,5]
        contourlevels = [x*rms for x in clevels]
        strs = [str(i)+'$\sigma$' for i in clevels]

    if clevel == 5:
        clevels = [-15,-10,-5,5,10,15]
        contourlevels = [x*rms for x in clevels]
        strs = [str(i)+'$\sigma$' for i in clevels]

    if clevel == 10:
        clevels = [-50,-40,-30,-20,-10,-5,5,10,20,30,40,50]
        contourlevels = [x*rms for x in clevels]
        strs = [str(i)+'$\sigma$' for i in clevels]

    if clevel == 3.5:
        clevels = [-8,-6,-5,-3,3,5,6,8]
        contourlevels = [x*rms for x in clevels]
        strs = [str(i)+'$\sigma$' for i in clevels]

    if clevel == 3.:
        clevels = [-12,-9,-6,-3,3,6,9,12]
        contourlevels = [x*rms for x in clevels]
        strs = [str(i)+'$\sigma$' for i in clevels]


    #Create figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    current_map = cm.get_cmap('magma')
    current_map.set_bad(color='black')
    im = ax.imshow(contSignal,origin='lower',extent=[xcnt[0],-1*xcnt[0],xcnt[0],-1*xcnt[0]],interpolation='bilinear',cmap=cm.magma,vmin=zmin,vmax=zmax)
    cont = ax.contour(contSignal,levels=contourlevels,colors='white',alpha=0.85,origin='lower',extent=[xcnt[0],-1*xcnt[0],xcnt[0],-1*xcnt[0]],linewidths=1.5)
    fmt = {}
    for l,s in zip(cont.levels,strs):
        fmt[l] = s
    ax.clabel(cont,cont.levels,inline=True,fmt=fmt,fontsize=20)
    cbar = plt.colorbar(im)
    cbar.set_label(barlabel,fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    #Add synthesized beam
    ax.add_patch(Ellipse((xcnt[0]-0.30*xcnt[0],xcnt[0]-0.30*xcnt[0]),width=bmaj,height=bmin,angle=bpa+90.,edgecolor='white',facecolor='none',hatch='///',linewidth=2))
    ax.tick_params(axis='x',direction='in',color='white',length=7,labelsize=20)
    ax.tick_params(axis='y',direction='in',color='white',length=7,labelsize=20)
    ax.tick_params(bottom=True,top=True,left=True,right=True)
    ax.tick_params(labelleft=True,labelbottom=True,labeltop=False,labelright=False)
    ax.set_xlabel('Distance West (km)',fontsize=22,color='black')
    ax.set_ylabel('Distance North (km)',fontsize=22)
    ax.set_title(figtitle,fontsize=24,fontweight='bold')

    #Add arrows for Sun and Tail along with illumination diagram if desired
    if (arrows==1):
        pwidth=xcnt[0]*0.20
        iwidth = pwidth*(2*ifrac-1)
        #Add illumination geometry
        ax.add_patch(Arc((-1*xcnt[0]+0.30*xcnt[0],xcnt[0]-0.30*xcnt[0]),height=pwidth,width=pwidth,theta1=90,theta2=270,color='yellow'))
        ax.add_patch(Arc((-1*xcnt[0]+0.30*xcnt[0],xcnt[0]-0.30*xcnt[0]),height=pwidth,width=iwidth,theta1=90,theta2=270,color='yellow'))
        ax.add_patch(Arc((-1*xcnt[0]+0.30*xcnt[0],xcnt[0]-0.30*xcnt[0]),height=pwidth,width=pwidth,theta1=270,theta2=90,color='yellow',linestyle='--'))
        #Solar vector
        #Convert psa and ps_amv to proper units
        # ps_amv = ps_amv*np.pi/180. + np.pi/2.
        # psa = psa*np.pi/180. + np.pi/2.
        # ax.annotate("",xytext=(-1*xcnt[0]+0.30*xcnt[0],xcnt[0]-0.30*xcnt[0]),xy=(-1*xcnt[0]+0.30*xcnt[0]+pwidth*np.cos(psa),xcnt[0]-0.30*xcnt[0]+pwidth*np.sin(psa)),xycoords='data',textcoords='data',arrowprops=dict(color='white',headwidth=10,width=0.1))
        # ax.text((-1*xcnt[0]+0.30*xcnt[0])+1.25*pwidth*np.cos(psa),(xcnt[0]-0.30*xcnt[0])+1.25*pwidth*np.sin(psa),'S',fontsize=18,color='white')
        # #Tail vector
        # ax.annotate("",xytext=(-1*xcnt[0]+0.30*xcnt[0],xcnt[0]-0.30*xcnt[0]),xy=(-1*xcnt[0]+0.30*xcnt[0]-1*pwidth*np.cos(ps_amv),xcnt[0]-0.30*xcnt[0]-1*pwidth*np.sin(ps_amv)),xycoords='data',textcoords='data',arrowprops=dict(color='white',headwidth=10,width=0.1))
        # ax.text((-1*xcnt[0]+0.30*xcnt[0])-1.15*pwidth*np.cos(ps_amv),(xcnt[0]-0.25*xcnt[0])-1.15*pwidth*np.sin(ps_amv),'T',fontsize=18,color='white')
        # Add sigma label
        ax.text((xcnt[0]-0.15*xcnt[0]),(-1*xcnt[0]+0.30*xcnt[0]),slabel,fontsize=16,color='white',fontweight='bold')
        # Change from NXR 1-22-2025
        r0 = np.abs(xcnt[0])
        ps_amv = ps_amv*np.pi/180 + np.pi/2
        psa = psa*np.pi/180 + np.pi/2 + np.pi
        ax.arrow(0.7*r0,-0.7*r0,0.2*r0*np.cos(psa),0.2*r0*np.sin(psa),head_width=0.05*r0,head_length=0.05*r0,facecolor='black')
        ax.arrow(0.7*r0,-0.7*r0,0.2*r0*np.cos(ps_amv),0.2*r0*np.sin(ps_amv),head_width=0.05*r0,head_length=0.05*r0,facecolor='black')

    plt.savefig(figname)
    plt.show()

def example():
    plot_alma('A3.cont.all.fixplanets.phaseshift.clean1.image.fits',3,1,264,227,0.26,0.743,'my.continuum.map.pdf','My Comets ### GHz Continuum',1000)
