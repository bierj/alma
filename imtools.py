import numpy as np

# Telescope beam area in sr    
def beamOmega(x,y):
    # x and y are the minor and major ellipse axes in arcsec
    # 4*pi*(180/pi)**2 *60**4 = 534638377800.0 is the number of square arcsec in a sphere
    # Uses formula for geometric area under a 2D Gaussian:
    
    return (4.*np.pi) * (2.*np.pi*x*y)/(534638377800.0*8.*np.log(2.0))

# Return sigma-clipped (masked) numpy array (iterate until new_RMS <= 1.01*previous_RMS). 
# posneg = 'positive': Vlaues > nSigma * RMS are masked
# posneg = 'negative': Vlaues < (-nSigma) * RMS are masked
# posneg = 'both' means both
def sigClip(img,nSigma,DELTARMS=1.01,posneg='both'):
    sigma = np.nanstd(img)
    sigma0 = sigma*2.
    imgmasked = img
    while sigma0/sigma > DELTARMS:
       if posneg == 'positive':
           imgmasked = np.ma.masked_where((img-np.nanmean(imgmasked))>nSigma*sigma,img)
       if posneg == 'negative':
           imgmasked = np.ma.masked_where((img-np.nanmean(imgmasked))<(-nSigma)*sigma,img)
       if posneg == 'both':
           imgmasked = np.ma.masked_where(np.abs(img-np.nanmean(imgmasked))>nSigma*sigma,img)       
       sigma0 = sigma
       sigma=np.nanstd(imgmasked)
       if imgmasked.mask.all(): print("All data points have been rejected, cannot continue")
    return imgmasked


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=1.0, weights=None, steps=False, interpnan=False, left=None, right=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        radial_prof = np.array([image.flat[whichbin==b].std() for b in range(1,nbins+1)])
    else:
        radial_prof = np.array([(image*weights).flat[whichbin==b].sum() / weights.flat[whichbin==b].sum() for b in range(1,nbins+1)])

    #import pdb; pdb.set_trace()

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    print(returnradii)

    if steps:
        xarr = np.array(list(zip(bins[:-1],bins[1:]))).ravel() 
        yarr = np.array(list(zip(radial_prof,radial_prof))).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof


def azimuthalAverage3d(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=1.0, weights=None, steps=False, interpnan=False, left=None, right=None):
    """
    Calculate the azimuthally spectrum as a function of radius.

    image - The 3D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image[:,:,0].shape)
    
    # Length of spectrum:
    splen = np.shape(image)[2]

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image[:,:,0].shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0
    
    specprofile = np.zeros([splen,nbins])

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        for i in range(splen):
            specprofile[i] = np.array([image[:,:,i].flat[whichbin==b].std() for b in range(1,nbins+1)])
    else:
        for i in range(splen):
            specprofile[i] = np.array([(image[:,:,i]*weights).flat[whichbin==b].sum() / weights.flat[whichbin==b].sum() for b in range(1,nbins+1)])

    profilespec=specprofile.transpose()

    if steps:
        xarr = np.array(list(zip(bins[:-1],bins[1:]))).ravel() 
        yarr = np.array(list(zip(radial_prof,radial_prof))).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,profilespec
    elif return_nr:
        return nr,bin_centers,profilespec
    else:
        return profilespec

def radialAverage(img, center=None, bintheta=1.0, binr=1.0, rmax=None):
    """
    Calculate the radially-averaged azimuthal profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels)
    bintheta - The angular bin size (in degrees)
    binr - Radial step
    """
    
    radAv = []

    from scipy.interpolate import RectBivariateSpline
    
    x = np.arange(0,len(img))
    y = np.arange(0,len(img[0]))
    
    if center is None:
        xcen, ycen = (x.max()-x.min())/2.0, (y.max()-y.min())/2.0
    else:
        xcen, ycen = center[0], center[1]
    
    if rmax is None:
       rmax = min(x.max()-xcen,xcen-x.min(),y.max()-ycen,ycen-y.min())
    
    print(rmax)
    
    iimage = RectBivariateSpline(x,y,img)
    
    rwalls = np.arange(0.,rmax,binr)
    r = (rwalls + binr/2)[:-1]
    
    thetas = list(range(0,360,bintheta))
    
    i=0
    for theta in thetas:
       radAv.append(0.0)
       for rad in r:
          x = xcen + (rad * np.cos(theta* np.pi/180.))
          y = ycen + (rad * np.sin(theta* np.pi/180.))
          print(iimage.ev(x, y)[0])
          radAv[i] += (iimage.ev(x, y)[0])
          print(radAv[i])
       i += 1
          
    return thetas, radAv


def thetaFluxFits(fitsimgname, r1,r2,ccw,bintheta=1.0,center=None,showPlots=False,error=False,getLats=False,subObsLat=0.):
    """
    Calculate the flux around a circle as a function of theta (summed along radial vector r1->r2).

    fitsimgename - The 2D fits image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels)
    bintheta - The angular bin size (in degrees)
    r1,r2 = radial integration range (in arcsec)
    ccw = angle of north pole (counter clockwise)
    """
    from scipy.interpolate import RectBivariateSpline
    from astropy.io import fits
    import pdb

    file = fits.open(fitsimgname)
    header = file[0].header
    
    subobslat = np.deg2rad(subObsLat)
    ccwrad = np.deg2rad(ccw)
    
    # Get image info
    xlen = header['NAXIS1']
    ylen = header['NAXIS2']
    # Pixel size
    platescale = abs(header['CDELT1']) * 3600
    #  Get the beam area in pixels 
    beam1 = header['BMAJ'] * 3600
    beam2 = header['BMIN'] * 3600
    px = header['CDELT1'] *np.pi/180
    py = header['CDELT2'] *np.pi/180
    pixbeam = beamOmega(beam1,beam2) / abs(px*py)
        
    x = np.arange(xlen)
    y = np.arange(ylen)
    
    if center:
        xcen = center[0]
        ycen = center[1]
    else:
        xcen = (xlen / 2.) -0.5
        ycen = (ylen / 2.) -0.5
    
    img = file[0].data
    
    #Coordinate grid WRT centre
    a,b=np.meshgrid(x-xcen,y-ycen) 
    
    #Polar coordinate grid
    theta=(np.pi-np.arctan2(a,-b))*180./np.pi + ccw
    thetamod = np.where(theta>=360.)
    theta[thetamod]=theta[thetamod]-360.
    
    r=(a**2+b**2)**0.5*platescale

    #Create latitudes for sphere at mid-radius; for higher altitudes use the lim latitude
    x=a*platescale
    y=b*platescale
    midrad=0.5*(r1+r2)
    northx = -np.sin(ccwrad)*np.cos(subobslat)
    northy = np.cos(ccwrad)*np.cos(subobslat)
    northz = np.sin(subobslat) 
    with np.errstate(divide='ignore',invalid='ignore'): 
        zcoord = np.sqrt(midrad**2 - x**2 - y**2)
        dprod = (northx*x + northy*y + northz*zcoord)/(midrad) #dot product of north pole vector and each vector in model planet
        z_lat = 90. - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid 
    zmask=np.where(np.isnan(z_lat))
    dprod = (northx*x + northy*y)/(r) #dot product of north pole vector and each vector in model planet
    z_lat_outside = 90. - np.degrees(np.arccos(dprod)) #latitude of limb below each x,y point
    z_lat[zmask]=z_lat_outside[zmask]

    #Create digitized theta array
    thetabin = np.arange(0,360+bintheta,bintheta)
    thetabincen = (thetabin-bintheta/2.)[1:]    
    thetabinin=np.digitize(theta,thetabin)-1   #This only works because theta is never outside the range of thetabin
    thetadigits=thetabincen[thetabinin]
    
    #Loop over theta bins - take average of all image pixels inside each theta bin (between r1 and r2)
    radSum=[]
    lats=[]
    npts=[]
    for theta in thetabincen:
        coords = np.where((thetadigits==theta)&(r>r1)&(r<r2))
        npts.append(len(img[coords]))
        radSum.append(np.mean(img[coords]))
        lats.append(np.ma.average(z_lat[coords],weights=img[coords])) 
    
    if showPlots:
        import matplotlib.pyplot as plt
        # Need to create a digitized radius array
        rbin = np.arange(r1,r2+platescale,platescale)
        rbincen = (rbin-platescale/2)[1:]    
        # Fake r values for 0th and len(rbin)th bins - this is needed because of the way digitize works with values off the ends of the range.
        rbincenplus = np.append(rbincen,1e20)
        rbincenplus = np.insert(rbincenplus, 0, 0.5*(rbin[0]))
        rbinin=np.digitize(r,rbin)
        rdigits=rbincenplus[rbinin]
        rdigitsm=np.ma.masked_where((r<r1)|(r>r2),rdigits)
        
        impolar=np.zeros((len(rbincen),len(thetabincen)))
        #Loop over the theta and r bins and find the pixels that match
        for theta,j in zip(thetabincen,list(range(len(thetabincen)))):
            for r,i in zip(rbincen,list(range(len(rbincen)))):
                coords = np.where((thetadigits==theta)&(rdigitsm==r))
                # Note: if this breaks, it may be because the image pixels are too large and no digitized r,theta values fall in the rbin,thetabin bins.
                impolar[i,j]=np.mean(img[coords])
        
        plt.ion()
        fig1=plt.figure()
        ax1=fig1.add_subplot(111)
        #ax1.imshow(impolar,origin='lower')
        ax1.pcolormesh(thetabincen,rbincen,impolar)
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Radius (arcsec)')
        
        fig2=plt.figure()
        ax2=fig2.add_subplot(111)
        ax2.plot(thetabincen,radSum,'k--')
        #ax2.fill_between(thetabincen,radSum-sigma,radSum+sigma)
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Average Flux (Jy/beam)')
    
    if getLats:        
       if error:
           #Get noise
           rms=np.std(sigClip(img,3))
           sigma = rms * (np.asarray(npts)/pixbeam)**-0.5 
           return thetabincen, np.asarray(radSum), sigma, lats
       else:      
           return thetabincen, np.asarray(radSum), lats
    else: 
       if error:
           #Get noise
           rms=np.std(sigClip(img,3))
           sigma = rms * (np.asarray(npts)/pixbeam)**-0.5 
           return thetabincen, np.asarray(radSum), sigma 
       else:      
           return thetabincen, np.asarray(radSum) 


def latFluxFits(fitsimgname, r1,r2,ccw,bintheta=1.0,center=None,showPlots=False,error=False,getLats=False,subObsLat=0.):
    """
    Calculate the flux around a circle as a function of latitude (summed along radial vector r1->r2).

    fitsimgename - The 2D fits image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels)
    bintheta - The angular bin size (in degrees)
    r1,r2 = radial integration range (in arcsec)
    ccw = angle of north pole (counter clockwise)
    """
    from scipy.interpolate import RectBivariateSpline
    from astropy.io import fits
    import pdb
    import matplotlib.pyplot as plt
    
    file = fits.open(fitsimgname)
    header = file[0].header
    
    ccw = np.deg2rad(ccw)
    subobslat = np.deg2rad(subObsLat)
    
    # Get image info
    xlen = header['NAXIS1']
    ylen = header['NAXIS2']
    # Pixel size
    platescale = abs(header['CDELT1']) * 3600
    #  Get the beam area in pixels 
    beam1 = header['BMAJ'] * 3600
    beam2 = header['BMIN'] * 3600
    px = header['CDELT1'] *np.pi/180
    py = header['CDELT2'] *np.pi/180
    pixbeam = beamOmega(beam1,beam2) / abs(px*py)
        
    x = np.arange(xlen)
    y = np.arange(ylen)
    
    if center:
        xcen = center[0]
        ycen = center[1]
    else:
        xcen = (xlen / 2.) -0.5
        ycen = (ylen / 2.) -0.5
    
    img = file[0].data
    
    #Coordinate grid WRT centre
    a,b=np.meshgrid(x-xcen,y-ycen)    
    
    #Rotate the coordinate frame
    arot=a*np.cos(-ccw)-b*np.sin(-ccw)
    brot=b*np.cos(-ccw)+a*np.sin(-ccw)
  
    #Polar coordinate grid
    theta=(np.arctan2(brot,np.abs(arot)))*180./np.pi

    r=(a**2+b**2)**0.5*platescale
     
    #Create latitudes for sphere at mid-radius; for higher altitudes use the lim latitude
    x=a*platescale
    y=b*platescale
    midrad=0.5*(r1+r2)
    northx = -np.sin(ccw)*np.cos(subobslat)
    northy = np.cos(ccw)*np.cos(subobslat)
    northz = np.sin(subobslat) 
    with np.errstate(divide='ignore',invalid='ignore'): 
        zcoord = np.sqrt(midrad**2 - x**2 - y**2)
        dprod = (northx*x + northy*y + northz*zcoord)/(midrad) #dot product of north pole vector and each vector in model planet
        z_lat = 90. - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid 
    zmask=np.where(np.isnan(z_lat))
    dprod = (northx*x + northy*y)/(r) #dot product of north pole vector and each vector in model planet
    z_lat_outside = 90. - np.degrees(np.arccos(dprod)) #latitude of limb below each x,y point
    z_lat[zmask]=z_lat_outside[zmask]
     
    #Create digitized theta array
    thetabin = np.arange(-90,90+bintheta,bintheta)
    thetabincen = (thetabin-bintheta/2.)[1:]    
    thetabinin=np.digitize(theta,thetabin)-1   #This only works because theta is never outside the range of thetabin
    thetadigits=thetabincen[thetabinin]

    #Loop over theta bins - take average of all image pixels inside each theta bin (between r1 and r2)
    radSum=[]
    lats=[]
    npts=[]
    for theta in thetabincen:
        coords = np.where((thetadigits==theta)&(r>r1)&(r<r2))
        npts.append(len(img[coords]))
        radSum.append(np.mean(img[coords]))
        lats.append(np.ma.average(z_lat[coords],weights=img[coords]))
    
    if showPlots:
        # Need to create a digitized radius array
        rbin = np.arange(r1,r2+platescale,platescale)
        rbincen = (rbin-platescale/2)[1:]    
        # Fake r values for 0th and len(rbin)th bins - this is needed because of the way digitize works with values off the ends of the range.
        rbincenplus = np.append(rbincen,1e20)
        rbincenplus = np.insert(rbincenplus, 0, 0.5*(rbin[0]))
        rbinin=np.digitize(r,rbin)
        rdigits=rbincenplus[rbinin]
        rdigitsm=np.ma.masked_where((r<r1)|(r>r2),rdigits)
        
        impolar=np.zeros((len(rbincen),len(thetabincen)))
        #Loop over the theta and r bins and find the pixels that match
        for theta,j in zip(thetabincen,list(range(len(thetabincen)))):
            for r,i in zip(rbincen,list(range(len(rbincen)))):
                coords = np.where((thetadigits==theta)&(rdigitsm==r))
                # Note: if this breaks, it may be because the image pixels are too large and no digitized r,theta values fall in the rbin,thetabin bins.
                impolar[i,j]=np.mean(img[coords])
        
        plt.ion()
        fig1=plt.figure()
        ax1=fig1.add_subplot(111)
        #ax1.imshow(impolar,origin='lower')
        ax1.pcolormesh(thetabincen,rbincen,impolar)
        ax1.set_xlabel('Latitude (degrees)')
        ax1.set_ylabel('Radius (arcsec)')
        
        fig2=plt.figure()
        ax2=fig2.add_subplot(111)
        ax2.plot(thetabincen,radSum,'k--')
        #ax2.fill_between(thetabincen,radSum-sigma,radSum+sigma)
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Average Flux (Jy/beam)')
    
    if getLats:    
       if error:
           #Get noise
           rms=np.std(sigClip(img,3))
           sigma = rms * (np.asarray(npts)/pixbeam)**-0.5 
           return thetabincen, np.asarray(radSum), sigma, lats 
       else:      
           return thetabincen, np.asarray(radSum), lats
    else:
       if error:
           #Get noise
           rms=np.std(sigClip(img,3))
           sigma = rms * (np.asarray(npts)/pixbeam)**-0.5 
           return thetabincen, np.asarray(radSum), sigma 
       else:      
           return thetabincen, np.asarray(radSum)


def radFluxWedgeFits(fitsimgname, r1,r2,theta1,theta2,ccw,center=None,showPlots=False,error=False):
    """
    Calculate the average flux as a function of radius, in an angular wedge (symmetric about celestial north) -- actually more of a bowtie shape

    fitsimgename - The 2D fits image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels)
    r1,r2 = radial range (in arcsec)
    theta1,theta2 = range of angles (0-180) in which to derive the radial profile - this range will be mirrored about north-south axis
    ccw is the angle of Titan's pole with respect to celestial north (in degrees)
    """
    from scipy.interpolate import RectBivariateSpline
    from astropy.io import fits
    import pdb
    
    ccw = np.deg2rad(ccw)

    file = fits.open(fitsimgname)
    header = file[0].header
    
    # Get image info
    xlen = header['NAXIS1']
    ylen = header['NAXIS2']
    # Pixel size
    platescale = abs(header['CDELT1']) * 3600
    #  Get the beam area in pixels 
    beam1 = header['BMAJ'] * 3600
    beam2 = header['BMIN'] * 3600
    px = header['CDELT1'] *np.pi/180
    py = header['CDELT2'] *np.pi/180
    pixbeam = beamOmega(beam1,beam2) / abs(px*py)
        
    x = np.arange(xlen)
    y = np.arange(ylen)
    
    if center:
        xcen = center[0]
        ycen = center[1]
    else:
        xcen = (xlen / 2.) -0.5
        ycen = (ylen / 2.) -0.5
    
    img = file[0].data
    
    #Coordinate grid WRT centre
    a,b=np.meshgrid(x-xcen,y-ycen) 
    
    #Rotate the coordinate frame
    arot=a*np.cos(-ccw)-b*np.sin(-ccw)
    brot=b*np.cos(-ccw)+a*np.sin(-ccw)
    
    #Polar coordinate grid
    theta=(np.pi-np.arctan2(arot,-brot))*180./np.pi
    r=(arot**2+brot**2)**0.5*platescale

    # Create a digitized radius array
    rbin = np.arange(r1,r2+platescale,platescale)
    rbincen = (rbin-platescale/2)[1:]    
    rbinin=np.digitize(r,rbin)
    # Fake r values for 0th and len(rbin)th bins - this is needed because of the way digitize works.
    rbincenplus = np.append(rbincen,1e20)
    rbincenplus = np.insert(rbincenplus, 0, 0.5*(rbin[0]))
    rdigits=rbincenplus[rbinin]
    rdigitsm=np.ma.masked_where((theta<theta1)|(theta>(360.-theta1))|((theta>theta2)&(theta<360.-theta2)),rdigits)
    
    avg=[]
    npts=[]
    # Find all theta values between theta1,theta2 and take average for each rdigitsm by looping over rbincen values
    for r in rbincen:
        coords=np.where(rdigitsm==r)
        npts.append(len(img[coords]))
        avg.append(np.mean(img[coords])) 
    
    if showPlots:
        import matplotlib.pyplot as plt
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(rbincen,avg,'k--')
        ax.set_xlabel('Radius (arcsec)')
        ax.set_ylabel('Average Flux (Jy/beam)')
        
        fig2=plt.figure()
        ax2=fig2.add_subplot(111)
        imgm=np.ma.masked_array(img,rdigitsm.mask)
        ax2.imshow(imgm,origin='lower')
        
    if error:
        #Get noise
        rms=np.std(sigClip(img,3))
        sigma = rms * (np.asarray(npts)/pixbeam)**-0.5    
        return np.asarray(rbincen), np.asarray(avg), sigma 
    else:      
        return np.asarray(rbincen), np.asarray(avg)
  
    
def gauss2d(xlen,ylen,fwhm_x,fwhm_y,dx,theta=0.,x0=0.,y0=0.,norm=True):
    #Returns the value of the 2d elliptical gaussian for grid of size xlen x ylen (pixels). Theta defined as angle in radians from x axis to semimajor axis (CCW). dx is the pixel scale (in same units as fwhm). x0 and y0 are offsets in units of dx
    
    x,y = np.indices([xlen,ylen])
    x = dx * (x-(x.max()-x.min())/2.0)
    y = dx * (y-(y.max()-y.min())/2.0)

    
    sig_x = fwhm_x / (2*np.sqrt(2*np.log(2))) #sig is the standard deviation
    sig_y = fwhm_y / (2*np.sqrt(2*np.log(2)))
    
    if norm:
        # Normalize the Gaussian to unit volume by pixel number
        A = 1. / (2.*np.pi*sig_x*sig_y/dx**2)
    else:
        A = 1.   
    
    # Rotated elliptical Gaussian
    a1 = np.cos(theta)**2/(2*sig_x**2) + np.sin(theta)**2/(2*sig_y**2)
    b1 = -np.sin(2*theta)/(4*sig_x**2) + np.sin(2*theta)/(4*sig_y**2)
    c1 = np.sin(theta)**2/(2*sig_x**2) + np.cos(theta)**2/(2*sig_y**2)
    g = A*np.exp(-(a1*(x-x0)**2 - 2*b1*(x-x0)*(y-y0) + c1*(y-y0)**2))
    return g


# Routine to extract spectrum from fits cube 'fitsFile' at specified x and y pixel position (zero-based)
# vel2freq converts velocity axis to frequencies, scaled by velfactor and freqfactor
def spExFits(fitsFile,xpix,ypix,vel2freq=False,freqfactor=1e-9,velfactor=1e-3):
   from astropy.io import fits as pyfits
   from scipy.constants import c
   hdu = pyfits.open(fitsFile)
   # Get image info
   cubeheader = hdu[0].header
   # Get the number of frequency channels
   nspec = cubeheader['NAXIS3']
   # reference freq in Hz
   f0 = float(cubeheader['CRVAL3']) 
   # channel width in Hz
   df = float(cubeheader['CDELT3']) 
   # reference pixel
   i0 = float(cubeheader['CRPIX3'])
 
   # Generate frequency grid
   specx = ((np.arange(nspec) - i0 +1) * df + f0) * velfactor
   # Do conversion to frequency (if requested)
   if vel2freq:
      specx = (1 - specx/c) * cubeheader['RESTFRQ'] * freqfactor / velfactor
   
   # Extract spectrum -- in CASA 4.0.1 imval can only extract regions, not individual pixels, but lower left pixel is in ['data'][0][0]
   specy = hdu[0].data[:,xpix,ypix]

   # Return arrays of frequency in GHz and flux    
   return specx,specy


 # Convolve a fits cube with a 2D beam pattern (numpy array); beamPixScale is in arcsec; drop allows the 0th image axis to be dropped, creating a 3D cube
def convolveFitsBP(fitsFile,beam,beamPixScale,drop=True):
   from astropy.io import fits as pyfits
   from scipy.interpolate import interp2d
   from scipy.signal import fftconvolve
   import sys,os
   fileout=fitsFile[:-5] + '_conv.fits'
   hdu = pyfits.open(fitsFile)
   cubeheader = hdu[0].header
   cube = hdu[0].data
   cubePixScale = cubeheader['CDELT1']*3600.
   
   # If it's a 4D cube, drop the zeroth (last) image dimension
   while (len(cube.shape)>3 and drop == True):
      print("Dropping zeroth axis of 4D fits image")
      cube = cube[0]
   
   #resample (and zero-pad) the beam
   beamx=(np.arange(0,beam.shape[0])-beam.shape[0]/2)*beamPixScale
   beami=interp2d(beamx,beamx,beam,kind='linear',bounds_error=False,fill_value=0.)
   nimgy = cubeheader['NAXIS1']
   nimgx = cubeheader['NAXIS2']
   
   imgx=(np.arange(0,nimgx)-(nimgx-1)/2.)*cubePixScale
   imgy=(np.arange(0,nimgy)-(nimgy-1)/2.)*cubePixScale
   ibeam=beami(imgx,imgy)
   
   #Do the convolution
   imgConv=np.zeros(cube.shape)
   for i in range(len(cube)):
      imgConv[i]=fftconvolve(cube[i],ibeam,mode='same')
      
   hdu = pyfits.PrimaryHDU(imgConv/ibeam.sum(), header = cubeheader)
   if os.path.exists (fileout):
      os.unlink (fileout)
   hdu.writeto(fileout)
   sys.stdout.write("Writing convolved cube to %s\n" % fileout)


#  Multiply 2D fits file by Gaussian beam shape

def pbMultiply(infits,outfits,hpbw):
    # hpbw is in arcsec
    from astropy.modeling.models import Gaussian2D
    from astropy import io
    
    c = 2.99792458e8

    fits = io.fits.open(infits)
    header = fits[0].header

    #Get the RA and DEC grid (centered at zero)
    naxis1 = header['naxis1']
    naxis2 = header['naxis2']

    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']

    ra = (np.arange(naxis1, dtype=np.double)-(naxis1/2.0)) * cdelt1 * 3600.
    dec = (np.arange(naxis2, dtype=np.double)-(naxis1/2.0)) * cdelt2 * 3600.

    #Get normalized Gaussian primary beam pattern for multiplication
    stdev = hpbw / (2 * np.sqrt (2 * np.log(2)))
    mod_ra,mod_dec = np.meshgrid(ra,dec)
    beam = Gaussian2D(amplitude=1,x_mean=0,y_mean=0,x_stddev=stdev,y_stddev=stdev)(mod_ra,mod_dec)

    pbdata = fits[0].data * beam

    hdu = io.fits.PrimaryHDU(pbdata, header=header)
    hdu.writeto(outfits,overwrite=True)


#  Multiply fits file (2D or 3D) by Gaussian beam shape and return the sum
def pbSum(infits,hpbw,drop=True,integrate=False):
    # hpbw is in arcsec
    from astropy.modeling.models import Gaussian2D
    from astropy import io
    
    c = 2.99792458e8

    fits = io.fits.open(infits)
    header = fits[0].header

    #Get the RA and DEC grid (centered at zero)
    naxis1 = header['naxis1']
    naxis2 = header['naxis2']

    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']

    ra = (np.arange(naxis1, dtype=np.double)-(naxis1/2.0)) * cdelt1 * 3600.
    dec = (np.arange(naxis2, dtype=np.double)-(naxis1/2.0)) * cdelt2 * 3600.

    #Get normalized Gaussian primary beam pattern for multiplication
    stdev = hpbw / (2 * np.sqrt (2 * np.log(2)))
    mod_ra,mod_dec = np.meshgrid(ra,dec)
    beam = Gaussian2D(amplitude=1,x_mean=0,y_mean=0,x_stddev=stdev,y_stddev=stdev)(mod_ra,mod_dec)

    if drop:
      data = fits[0].data[0]
    else:
      data = fits[0].data


    pbdata = data * beam
    sp = np.squeeze(np.sum(pbdata,axis=(1,2)))

    if integrate:
      df = header['cdelt3'] # Hopefully the channel width in m/s.
      return np.sum(sp) * df * 1e-17 * header['RESTFRQ']/1e9 / c # Integrated flux in W/m^2 assuming input in Jy
    else:
      return sp
    


# Multiply fits cube by Gaussian beam and extract the summed (beam-weighted) spectrum

def convImgSpec(infits,hpbwx,hpbwy,theta,eta,units,xoff=0,yoff=0,freqAx=False,drop=True,outfile=None,integrate=False):
   # 1D Spectrum is extracted from a 3D image cube after applying a Gaussian beam shape (hpbwx,hpbwy are the major and minor axes in arcsec, theta is the position angle in degrees (CCW from the x axis), and xoff,yoff are the beam offset positions in arcsec). If drop=True, then only the first stokes axis of the cube is used. This (multiplicative) algorithm is much faster than a full convolution, but only applies for images in flux/pixel units. If the image units were Kelvin (surface brightness), we normalize the Gaussian so that the beam sum becomes an average.
   from scipy.constants import c
   from astropy.io import fits
   from pymodules.io import write2col
   import sys   
      
      
   f = fits.open(infits)
   header = f[0].header
        
   naxis1 = header['NAXIS1']
   naxis2 = header['NAXIS2']

   cdelt1 = header['CDELT1']

   if drop:
      data = f[0].data[0]
   else:
      data = f[0].data

   if units == 0:
      norm = True  # Kelvin scale
   else: 
      norm = False # Jansky scale

   g=gauss2d(naxis1,naxis2,hpbwx,hpbwy,abs(cdelt1*3600),theta=np.deg2rad(theta+90),x0=xoff,y0=yoff,norm=norm)
   
   dg = data * g
   spectrum = np.sum(dg,axis=(1,2)) * eta
     
   # Generate frequency grid
   nspec = len(spectrum)
   f0 = header['CRVAL3']
   df = header['CDELT3']/1e3
   i0 = header['CRPIX3']
   xgrid = ((np.arange(nspec) - i0 + 1) * df + f0)
 
   if freqAx:
     xgrid = (1. - (xgrid/(c/1e3))) * (header['RESTFRQ'])

   if outfile:
      write2col(xgrid,spectrum,outfile)

   if integrate:
   #Get integrated intensity
      Tdv = sum(spectrum) * df
      sys.stdout.write ( "Integrated intensity = %.3f\n" % Tdv)


   return xgrid, spectrum
