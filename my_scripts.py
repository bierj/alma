import math
import shutil
from copy import copy
from pathlib import Path
from pprint import pprint

import casatasks
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import scipy.optimize
import scipy.stats

import vis_sample
from analysis_scripts import analysisUtils as au
from line_modeling import sublimed1dFit2 as sublime2
from line_modeling import sublimed1dFit3 as sublime3
from plot_alma_auto import plot_alma
from vis_sample import Visibility

# autopep8:off
import sys # isort:skip
sys.path.append('/home/zedd/sharedcode')
import myphy as my # isort:skip
from myphy import Q # isort:skip
from myphy import units # isort:skip
# autopep8:on

CODEDIR = Path('/home/zedd/alma/code')
ROOTDIR = Path('/home/zedd/alma')
VIS_ACA = Path('/home/zedd/alma/ACA/uid___A002_X11e3e46_X6e92.ms.A3')
VIS_12M = Path('/home/zedd/alma/12m/uid___A002_X11e3e46_X6f09.ms.A3')
VIS_12M_CONT = Path(str(VIS_12M) + '.cont')


def linterp(x, xlo, xhi, ylo, yhi):
    t = (x - xlo) / (xhi - xlo)
    return t * ylo + (1 - t) * yhi


def getcellsize(vis):
    return au.pickCellSize(str(vis), imsize=True)


def getrefval(image):
    header = casatasks.imhead(str(image))
    ha = my.HourAngle(header['refval'][0])
    dec = my.Declination(header['refval'][1])
    return f'J2000 {str(ha)} {str(dec)}'


def getmaxposf(image):
    stats = casatasks.imstat(str(image))
    maxposf = [x.strip() for x in stats['maxposf'].split(',')]
    return f'J2000 {maxposf[0]} {maxposf[1]}'


def split_comet_A3(vis):
    out = str(vis) + '.A3'
    casatasks.split(vis=str(vis),
                    outputvis=out,
                    field='2',
                    spw='17,25,27,29',
                    keepflags=False)


def split_continuum(vis, outname, width):
    out = str(vis) + '.cont'
    casatasks.split(vis=str(vis), outputvis=out, field='0',
                    datacolumn='data', spw='0,3', width=width)
    vis = out
    cell, imsize, _ = getcellsize(vis)
    cell = f'{cell}arcsec'
    out = vis.parent / outname
    casatasks.tclean(vis=str(vis),
                     imagename=str(out),
                     field='0',
                     spw='0~1',
                     cell=cell,
                     imsize=imsize,
                     phasecenter='TRACKFIELD',
                     restoringbeam='common')


def split_aca_continuum():
    split_continuum(VIS_ACA, 'A3.cont.dirty', width=[16, 480])


def split_12m_continuum():
    split_continuum(VIS_12M, 'A3.cont.dirty', width=[16, 480])


def tclean_continuum(vis, image, outname, maskpath, cell, imsize, noise_rms=None):
    threshold = f'{2*noise_rms}Jy'
    cell, imsize, _ = getcellsize(vis)
    cell = f'{cell}arcsec'
    out = image.parent / outname
    casatasks.tclean(str(image),
                     imagename=str(out),
                     field='0',
                     spw='0~1',
                     cell=cell,
                     imsize=imsize,
                     phasecenter='TRACKFIELD',
                     restoringbeam='common',
                     niter=5e4,
                     threshold=threshold,
                     mask=str(maskpath),
                     pbcor=True)


def tclean_aca_continuum():
    cell, imsize, _ = getcellsize(VIS_ACA)
    image = ROOTDIR / 'A3.cont.dirty.image'
    outname = 'A3.cont.clean'
    maskpath = VIS_ACA.parent / 'comet.region'
    tclean_continuum(VIS_ACA, image, outname, maskpath, cell, imsize)


def tclean_12m_continuum():
    cell, imsize, _ = getcellsize(VIS_12M)
    image = ROOTDIR / 'A3.cont.dirty.image'
    outname = 'A3.cont.clean'
    maskpath = VIS_12M.parent / 'comet.region'
    tclean_continuum(VIS_12M, image, outname, maskpath, cell, imsize)


def center_continuum(vis, image, outname):
    cell, imsize, _ = getcellsize(vis)
    tclean_continuum(vis, image, outname, cell/100, [100*i for i in imsize])
    refval = getrefval(image)
    maxposf = getmaxposf(image)
    # direction = 'J2000 11:08:26.014 -05.38.56.524'
    # phasecenter = 'J2000 11:08:25.52360 -05.39.00.11430'
    fix = vis.parent / (vis.name + '.fixplanets')
    shutil.copytree(vis, fix)
    casatasks.fixplanets(vis=fix, field='0', fixuvw=True, direction=refval)
    phaseshift = vis.parent / (vis.name + '.fixplanets.phaseshift')
    casatasks.phaseshift(vis=fix, outputvis=phaseshift, phasecenter=maxposf)
    tclean_continuum(vis=phaseshift,
                     image=str(vis.parent / outname),
                     cell=cell,
                     imsize=imsize)


def plot_continuum(imagepath, fitspath, figpath, figtitle):
    casatasks.exportfits(imagename=imagepath,
                         fitsimage=fitspath,
                         velocity=True,
                         dropdeg=True,
                         dropstokes=True)
    plot_alma(contfile=imagepath,
              clevel=3,
              arrows=1,
              psa=264,
              ps_amv=227,
              ifrac=0.26,
              delta=0.743,
              figname=figpath,
              figtitle=figtitle,
              prange=1000)


def split_12m_timebinned_continuum():
    # cont should already have 8 chans per SPW so no need to set width for SPW1 (3 from orig.ms)
    in_ = str(VIS_12M_CONT) + '.fixplanets.phaseshift'
    out = str(in_) + '.SPW0.timebin'
    casatasks.split(vis=str(in_),
                    outputvis=out,
                    datacolumn='data',
                    spw='0',
                    keepflags=False,
                    timebin='1e6s',
                    combine='scan')
    out = str(in_) + '.SPW3.timebin'
    casatasks.split(vis=str(in_),
                    outputvis=out,
                    datacolumn='data',
                    spw='1',
                    keepflags=False,
                    timebin='1e6s',
                    combine='scan')


def bin_visibilities(vis, width, offset, uvmax):
    f0 = Q(0.5*(vis.freqs[0] + vis.freqs[-1]), 'Hz')
    # baseline length in uv coords
    uv = ((vis.vv**2 + vis.uu**2)**0.5 * my.c / f0).to('m')
    vis_RMS = Q(np.nanstd(np.real(vis.VV)), 'Jy')
    VV = vis.VV[:]

    if uvmax is not None:
        i = np.where(uv <= uvmax)
        VV = VV[i]
        uv = uv[i]

    bins = np.arange(offset.magnitude, np.max(uv).magnitude, width.magnitude)
    bin_inds = np.digitize(uv.magnitude, bins)

    rms = np.zeros(len(uv)) + vis_RMS

    uv_binned = []
    VV_binned = []
    amp_binned = []
    re_binned = []
    im_binned = []
    err = []
    for i in range(len(bins)):
        i_bins = bin_inds == i
        if np.sum(i_bins) > 0:
            uv_binned.append(np.average(uv[i_bins].magnitude))
            VV_avg = np.average(VV[i_bins])
            VV_binned.append(VV_avg)
            re_binned.append(np.real(VV_avg))
            im_binned.append(np.imag(VV_avg))
            amp_binned.append(np.abs(VV_avg))
            # if visibilities are over more than one channel, it should be the last dimension
            if len(VV.shape) > 1:
                err.append(1.29 / np.sum(VV.shape[-1] / rms[i_bins]**2)**0.5)
            else:
                err.append(1.29 / np.sum(1 / rms[i_bins]**2)**0.5)

    return {'bins': bins,
            'uv': Q(np.asarray(uv_binned), 'm'),
            'VV': np.asarray(VV_binned),
            'amp': Q(np.asarray(amp_binned), 'Jy'),
            're': Q(np.asarray(re_binned), 'Jy'),
            'im': Q(np.asarray(im_binned), 'Jy'),
            'err': Q(np.asarray([e.magnitude for e in err]), 'Jy')}


def scale_visibilities(vis, f_common, spectral_index):
    scaled_vis = copy(vis)
    scaled_vis.VV = vis.VV[:]
    for i, f in enumerate(vis.freqs):
        s = float(f_common / Q(f, 'Hz'))**spectral_index
        scaled_vis.VV[:, i] *= s
    return scaled_vis


def append_visibilities(*viss):
    if len(viss) < 1:
        return None
    v0 = viss[0]
    if len(viss) == 1:
        return v0
    VV = v0.VV
    freqs = v0.freqs
    for vis in viss[1:]:
        VV = np.append(VV, vis.VV, axis=1)
        freqs = np.append(freqs, vis.freqs)
    return Visibility(VV,
                      v0.uu,
                      v0.vv,
                      v0.wgts,
                      freqs,
                      np.zeros_like(v0.uu))


def append_then_bin_visibilities(f_common, spectral_index,
                                 *viss,
                                 binoffset=Q(0, 'm'),
                                 binwidth=Q(8, 'm'),
                                 uvmax=Q(400, 'm')):
    scaled = [scale_visibilities(vis, f_common, spectral_index)
              for vis in viss]

    all_vis = append_visibilities(*scaled)

    # width shouldn't be smaller than 6 (half antenna separation)
    # uvmax to filter out longer baselines with high SNR
    return bin_visibilities(all_vis, binwidth, binoffset, uvmax)


def plot_visibilities(vis):
    _, ax = plt.subplots(figsize=(15, 10))

    ax.errorbar(vis['uv'], vis['re'], vis['err'], marker='o',
                linestyle=' ', markersize=3)

    ax.set_ylabel('Real (Jy)')
    ax.set_xlabel('uv-distance (m)')

    plt.title('Visibilities')
    plt.show()


def dust_model_curve_fit():
    vis_spw0 = str(VIS_12M) + '.cont.fixplanets.phaseshift.SPW0.timebin'
    vis_spw3 = str(VIS_12M) + '.cont.fixplanets.phaseshift.SPW3.timebin'

    spw0 = vis_sample.import_data_ms(vis_spw0)
    spw3 = vis_sample.import_data_ms(vis_spw3)

    f_common = Q(459, 'GHz')

    # should fit this ourselves to confirm since this was fitted from data ~200 GHz (Lellouch+2022, A&A, 659, L1)
    # fit with spw without averaging channels (SPW3 = 3840 chans)
    # possibly fit from Spitzer studies
    # look up prev studies of comet continuum
    spectral_index = 1.93

    fig, axs = plt.subplots(3, 3, sharex='col', figsize=(15, 15 * 9 / 16))
    fig.suptitle('Visibilities')
    fig.supylabel('Real (Jy)')
    fig.supxlabel('uv-distance (m)')

    for i in range(0, 9):
        width = Q(2 * i + 10, 'm')

        vis = append_then_bin_visibilities(f_common, spectral_index, spw0, spw3,
                                           binwidth=width)

        (a, b, c), pconv = scipy.optimize.curve_fit(
            lambda x, a, b, c: a*x**b + c,
            vis['uv'].m, vis['re'].m,
            [1, -1, 1e-3])

        np.savetxt(ROOTDIR / '12m' / 'dust_curve_fit' /
                   f'pconv_{width.m}_m.txt', pconv)

        row = i // 3
        col = i - row * 3
        ax = axs[row, col]

        ax.errorbar(vis['uv'].m, vis['re'].m, vis['err'].m, marker='o',
                    linestyle=' ', markersize=3, label='Re(VV)')

        fitx = np.arange(12, 401)
        fity = [a*x**b + c for x in fitx]
        ax.plot(fitx, fity, label=f'Re(VV) = {a:.2g} uv^{b:.2g} + {c:.2e}')

        ax.legend()
        ax.set_title(f'width = {width}')
    outfile = ROOTDIR / '12m' / 'dust_curve_fit' / 'visibilities.png'
    fig.savefig(outfile)


def dust_model_lmfit():
    vis_spw0 = str(VIS_12M) + '.cont.fixplanets.phaseshift.SPW0.timebin'
    vis_spw3 = str(VIS_12M) + '.cont.fixplanets.phaseshift.SPW3.timebin'

    spw0 = vis_sample.import_data_ms(vis_spw0)
    spw3 = vis_sample.import_data_ms(vis_spw3)

    f_common = Q(459, 'GHz')

    # should fit this ourselves to confirm since this was fitted from data ~200 GHz (Lellouch+2022, A&A, 659, L1)
    # fit with spw without averaging channels (SPW3 = 3840 chans)
    # possibly fit from Spitzer studies
    # look up prev studies of comet continuum
    spectral_index = 1.93

    binwidth = Q(24, 'm')

    vis = append_then_bin_visibilities(
        f_common, spectral_index, spw0, spw3, binwidth=binwidth)

    outfile = VIS_12M.parent / 'dust_lmfit' / f'visibilities_{binwidth.m}m.txt'
    np.savetxt(outfile, np.vstack((vis['uv'].m, vis['re'].m)).T)

    fig, ax = plt.subplots(1, 1, sharex='col', figsize=(15, 15 * 9 / 16))
    fig.suptitle(f'Binned Visibilities, width = {binwidth}')
    fig.supylabel('Real (mJy)')
    fig.supxlabel('uv-distance (m)')
    ax.errorbar(vis['uv'].m, vis['re'].m * 1e3, vis['err'].m * 1e3,
                marker='o', linestyle=' ', color='k')

    scenarios = ['point', 'coma', 'point+coma']
    colors = ['r', 'g', 'b']

    for i, scenario in enumerate(scenarios):
        model = None

        def getChisq(params):
            nonlocal model

            getChisq.ncalls += 1

            params.pretty_print(oneline=False, precision=5,
                                fmt='e', columns=['value', 'vary'])

            # Generate the model
            # For a point-source, V(sigma) = pointFlux
            # For a 1/rho coma, V(sigma) = K * c / (sigma * nu) = K * lambda / sigma
            # TODO: find ref
            # pointFlux, comaFlux = p
            model = np.asarray(
                [(params['pointflux'] + (my.c / f_common).m * params['comaflux'] / r.m) for r in vis['uv']])

            residuals = np.ravel((vis['re'].m - np.real(model)) /
                                 vis['err'].m[:, np.newaxis])

            print('Chisq ({:d} = '.format(getChisq.ncalls),
                  np.sum(residuals**2), "\n")
            return residuals
        getChisq.ncalls = 0

        params = lmfit.Parameters()
        if scenario == 'point':
            params.add('pointflux', value=Q(1e-5, 'Jy').m, vary=True)
            params.add('comaflux', value=Q(0, 'Jy').m, vary=False)
        elif scenario == 'coma':
            params.add('pointflux', value=Q(0, 'Jy').m, vary=False)
            params.add('comaflux', value=Q(1e-4, 'Jy').m, vary=True)
        else:
            params.add('pointflux', value=Q(1e-5, 'Jy').m, vary=True)
            params.add('comaflux', value=Q(1e-4, 'Jy').m, vary=True)

        print('*****************************************')
        print(f"Starting LMFIT for {scenario} scenario...")

        chisqtol = .01
        xtol = 1e-4
        ftol = chisqtol / len(vis['re'])

        result = lmfit.minimize(getChisq, params, xtol=xtol, ftol=ftol,
                                epsfcn=1e-5, scale_covar=False)

        pf = result.params

        # Final run with optimized parameters to generate best-fit file and plots
        print()
        print('Final call to function...')
        getChisq(pf)

        print('LMFIT exit: ', result.message)

        print('Number of calls to getChisq: ', getChisq.ncalls)

        # Reduced chi square
        print('\nReduced Chisq X_r = %.5f' % (result.redchi))
        print('P (probability that model is different from data due to chance) = %.3f\n' %
              scipy.stats.chi2.sf(result.chisqr, result.nfree))

        pfsigmas = {}
        for parname in params.keys():
            try:
                pfsigmas[parname] = pf[parname].stderr + 0.
            except:
                pfsigmas[parname] = 0.

        print('\nBest-fit parameters and 1-sigma covariance errors')
        print(
            'Point-Flux = {:8.4e} +- {:8.4e}'.format(pf['pointflux'].value, pfsigmas['pointflux']))
        print(
            'Power-Flux = {:8.4e} +- {:8.4e}'.format(pf['comaflux'].value, pfsigmas['comaflux']))

        outfile = VIS_12M.parent / 'dust_lmfit' / f'modelparams_{scenario}.txt'
        f = open(outfile, 'w')
        f.write("LMFIT Results: " + result.message)
        f.write('\nChi-Squre = {:.8f}\n'.format(result.chisqr))
        f.write('\n Reduced Chisq X_r = {:.8f}\n'.format(result.redchi))
        f.write('\nBest-fit parameters and 1-sigma covariance errors:\n')
        f.write('Point-Flux = {:8.4e} +- {:8.4e}\n'.format(
            pf['pointflux'].value, pfsigmas['pointflux']))
        f.write(
            'Power-Flux = {:8.4e} +- {:8.4e}\n'.format(pf['comaflux'].value, pfsigmas['comaflux']))
        f.close()
        print(f"Model run parameters written to {outfile}")
        print('*****************************************')
        print()

        outfile = VIS_12M.parent / 'dust_lmfit' / f'model_{scenario}.txt'
        np.savetxt(outfile, np.vstack((vis['uv'].m, model)).T)

        ax.plot(vis['uv'].m, np.real(model) * 1e3,
                color=colors[i], label=scenario)
        ax.legend()

    outfile = VIS_12M.parent / 'dust_lmfit' / 'fits.png'
    fig.savefig(outfile)


def bin_and_report_lowest_baseline_A3():
    vis_spw0 = str(VIS_12M) + '.cont.fixplanets.phaseshift.SPW0.timebin'
    vis_spw3 = str(VIS_12M) + '.cont.fixplanets.phaseshift.SPW3.timebin'

    spw0 = vis_sample.import_data_ms(vis_spw0)
    spw3 = vis_sample.import_data_ms(vis_spw3)

    vis = append_then_bin_visibilities(Q(459, 'GHz'), 1.93, spw0, spw3,
                                       bin_width=Q(12, 'm'))

    print(f'uv = {vis["uv"][0]}')
    print(f're = {vis["re"][0]:e}')
    print(f'im = {vis["im"][0]:e}')
    print(f'amp = {vis["amp"][0]:e}')
    print(f'err = {vis["err"][0]:e}')


def nucleus_temp(theta, phi, T0, alpha):
    if theta < alpha - my.PIOVERTWO or theta > alpha + my.PIOVERTWO:
        return Q(0, 'K')
    else:
        return T0 * math.cos(phi)**.25 * math.cos(theta)**.25


def subsolar_temp(albedo, emissitivity, beaming_factor, heliocentric_distance):
    numer = (1 - albedo) * my.solar_constant
    denom = emissitivity * my.sigma_sb * beaming_factor * heliocentric_distance**2
    return ((numer / denom)**.25).to('K')


def subsolar_temp_A3():
    albedo = .0212
    emissitivity = .9
    beaming_factor = 1.175
    heliocentric_distance = 0.405766181926  # AU
    return subsolar_temp(albedo, emissitivity, beaming_factor, heliocentric_distance)


def nucleus_diameter(flux, wavelength, alpha, emissitivity, geocentric_distance, report_err=False):
    T0 = subsolar_temp_A3()

    def integrand(theta, phi):
        a = math.cos(phi)**2 * math.cos(theta - alpha)
        T = nucleus_temp(theta, phi, T0, alpha)
        if T == 0:
            return 0
        else:
            b = math.exp(my.h * my.c / (wavelength * my.kb * T))
            return 1 / (b - 1.) * a
    # blackbody integral
    bb, bb_err = scipy.integrate.dblquad(integrand,
                                         0, my.PIOVERTWO,
                                         -my.PIOVERTWO, my.PIOVERTWO)

    print(f"bb, bb_err = {bb}, {bb_err}")

    flux_ = flux / wavelength * my.photon_frequency(wavelength)
    numer = flux_ * geocentric_distance**2 * wavelength**5
    denom = emissitivity * my.h * my.c**2 * float(bb)
    diameter = numer / denom
    if report_err:
        return (diameter**.5).to('km'), bb_err
    else:
        return (diameter**.5).to('km')


# average spatial profiles to see if anything pops out
# immath, mode='evalexpr', expr = '(IM0+IM1+IM2)/3'
# cell shape and size are the same
# same # of channels, channels one-to-one, same x-axis
# moment map - immoments, outfile = imagename+'.mom0'
# especially for disks showing spatial distribution for first moment

# Table 2 from Roth, 2021


# eta ~ .9 - 1.175
# radio emissitivity ~ .6 - .8
def nucleus_diameter_A3(flux, report_err=False):
    wavelength = my.photon_wavelength(Q(459, 'GHz'))
    alpha = Q(118.7183, 'deg').to('rad')
    emissitivity = .7
    geocentric_distance = Q(0.74056486704488, 'AU')  # from JPL Horizons
    return nucleus_diameter(flux, wavelength, alpha, emissitivity, geocentric_distance, report_err=report_err)


def nucleus_diameter_A3_curve_fit():
    # flux = Q(9.92e-4, 'Jy') + Q(5.34e-07, 'Jy^2')**.5 * sigma
    flux = Q(9.92e-4, 'Jy').plus_minus(Q(7.307530e-04, 'Jy'))
    return nucleus_diameter_A3(flux)
    # >>> my.nucleus_diameter_A3_curve_fit()
    # <Measurement(5.677071164261972, 2.090996363154198, kilometer)>


def nucleus_diameter_A3_lmfit():
    flux = Q(5.5731e-04, 'Jy').plus_minus(Q(1.2299e-04, 'Jy'))
    return nucleus_diameter_A3(flux)
    # >>> my.nucleus_diameter_A3_lmfit()
    # <Measurement(4.255170046040206, 0.4695262636257063, kilometer)>


def dust_mass(flux, wavelength, albedo, heliocentric_distance, geocentric_distance, opacity):
    dust_grain_temp = Q(277 * (1 - albedo)**.25 / heliocentric_distance.m**.5, 'K')
    numer = flux * wavelength**2 * geocentric_distance**2
    denom = 2 * my.kb * dust_grain_temp * opacity
    return (numer / denom).to('kg')


def dust_mass_A3(flux, wavelength, opacity):
    albedo = .0212
    heliocentric_distance = Q(0.405766181926, 'AU')
    geocentric_distance = Q(0.74056486704488, 'AU')
    return dust_mass(flux, wavelength, albedo, heliocentric_distance, geocentric_distance, opacity)


def run_my_NEATM_A3():
    print(f'Subsolar temperature = {subsolar_temp_A3()}')

    diam = nucleus_diameter_A3_lmfit()
    print(f'A3 point+coma fit diameter: {diam}')

    flux = Q(6.037036e-03, 'Jy').plus_minus(Q(2.152269e-03, 'Jy'))
    
    wavelength = my.photon_wavelength(Q(459, 'GHz')).to('mm')

    opacities = [
        {'lambda': Q(.45, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 0,
         'opacity': Q(.274, 'm^2 / kg')},
        {'lambda': Q(.45, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 22,
         'opacity': Q(.300, 'm^2 / kg')},
        {'lambda': Q(.45, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 48,
         'opacity': Q(.323, 'm^2 / kg')},
        {'lambda': Q(.8, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 0,
         'opacity': Q(.206, 'm^2 / kg')},
        {'lambda': Q(.8, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 22,
         'opacity': Q(.205, 'm^2 / kg')},
        {'lambda': Q(.8, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 48,
         'opacity': Q(.199, 'm^2 / kg')}
    ]

    lambda_lo = Q(.45, 'mm')
    lambda_hi = Q(.8, 'mm')
    kappa_lo = {0: opacities[0]['opacity'],
                22: opacities[1]['opacity'],
                48: opacities[2]['opacity']}
    kappa_hi = {0: opacities[3]['opacity'],
                22: opacities[4]['opacity'],
                48: opacities[5]['opacity']}
    opacities.append({'lambda': wavelength,
                      'size index': -3,
                      'max size': Q(10, 'mm'),
                      'porosity': 50,
                      'ice fraction': 0,
                      'opacity': linterp(wavelength, lambda_lo, lambda_hi, kappa_lo[0], kappa_hi[0])})
    opacities.append({'lambda': wavelength,
                      'size index': -3,
                      'max size': Q(10, 'mm'),
                      'porosity': 50,
                      'ice fraction': 22,
                      'opacity': linterp(wavelength, lambda_lo, lambda_hi, kappa_lo[22], kappa_hi[22])})
    opacities.append({'lambda': wavelength,
                      'size index': -3,
                      'max size': Q(10, 'mm'),
                      'porosity': 50,
                      'ice fraction': 48,
                      'opacity': linterp(wavelength, lambda_lo, lambda_hi, kappa_lo[48], kappa_hi[48])})

    def mass(opacity):
        return dust_mass_A3(flux, wavelength, opacity)
    masses = [dict(op, mass=mass(op['opacity'])) for op in opacities]
    print(f'{"lambda":>10} {"sz ind":>8} {"max sz":>8} {"porosity":>10} {"ice frac":>10} {"opacity":>14} {"mass":>12}')
    print(f'{"(mm)":>10} {"":>8} {"(mm)":>8} {"(%)":>10} {"(%)":>10} {"(m^2/kg)":>14} {"(kg)":>12}')
    for m in masses:
        print(f'{m["lambda"].m:>10.2} {m["size index"]:>8} {m["max size"].m:>8} {m["porosity"]:>10} {m["ice fraction"]:>10} {m["opacity"].m:>14.3} {m["mass"].m:>12.1e}')
# >>> my.run_my_NEATM_A3()
# Subsolar temperature = 606.0759753484152 kelvin
# bb, bb_err = 7.569885474802711, 1.4449154583928703e-07
# A3 point+coma fit diameter: (4.3 +/- 0.5) kilometer
#     lambda   sz ind   max sz   porosity   ice frac        opacity         mass
#       (mm)              (mm)        (%)        (%)       (m^2/kg)         (kg)
#       0.45       -3       10         50          0          0.274      9.7e+07+/-     3.4e+07
#       0.45       -3       10         50         22            0.3      8.8e+07+/-     3.1e+07
#       0.45       -3       10         50         48          0.323      8.2e+07+/-     2.9e+07
#        0.8       -3       10         50          0          0.206      1.3e+08+/-     0.5e+08
#        0.8       -3       10         50         22          0.205      1.3e+08+/-     0.5e+08
#        0.8       -3       10         50         48          0.199      1.3e+08+/-     0.5e+08
#       0.65       -3       10         50          0          0.245      1.1e+08+/-     0.4e+08
#       0.65       -3       10         50         22           0.26      1.0e+08+/-     0.4e+08
#       0.65       -3       10         50         48          0.271      9.8e+07+/-     3.5e+07


def run_Roth_NEATM_A3():
    import astropy.units as u
    from uncertainties import ufloat

    from NEATM import neatm

    # albedo = pv * phase_int
    phase_int = 0.4
    pv = 0.053
    # albedo = 0.0212
    rad_emissivity = 0.7
    bolo_emissivity = 0.9
    eta = 1.175  # beaming factor
    objectname = 'C/2023 A3'
    obsdate = '2024-10-01'
    obstime = '15:30:00'
    nu = 459 * u.GHz

    neatm_A3 = neatm(objectName=objectname, obsDate=obsdate, obsTime=obstime, nu=nu,
                     pv=pv, phase_int=phase_int, bolo_emissivity=bolo_emissivity,
                     rad_emissivity=rad_emissivity, eta=eta, uncertainties=False)

    # Report blackbody integral value and error
    print(f"bb, bb_err = {neatm_A3.bb[0]}, {neatm_A3.bb[1]}")

    # Calculate nucleus diameters for best fits to point-source fluxes
    flux = ufloat(5.5731e-04, 1.2299e-04) * (1 * u.Jy)
    diam = neatm_A3.calcDiameter(flux=flux)
    print(f'A3 point+coma fit diameter: {diam}')

    # uv = 17.99 m
    # re = 6.037036e-03 Jy
    # im = 1.408125e-03 Jy
    # amp = 6.199082e-03 Jy
    # err = 2.152269e-03 Jy

    # opacities:
    # astro silicates,
    # wavelength = .8mm and .45mm,
    # size index = -3,
    # max size = 10mm,
    # porosity = 50%,
    # ice fraction = 0%, 22%, 48%
    flux = ufloat(6.037036e-03, 2.152269e-03) * (1 * u.Jy)
    wavelength = my.photon_wavelength(Q(459, 'GHz')).to('mm')
    # heliocentric_distance = Q(0.405766181926, 'AU')

    # hpbwx = Q(0.48520079255104065, 'arcseconds').m
    # hpbwy = Q(0.4330733120441437, 'arcseconds').m
    # phi_synth = (hpbwx**2 + hpbwy**2)**.5 * u.arcsec

    opacities = [
        {'lambda': Q(.45, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 0,
         'opacity': .274 * u.m * u.m / u.kg},
        {'lambda': Q(.45, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 22,
         'opacity': .300 * u.m * u.m / u.kg},
        {'lambda': Q(.45, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 48,
         'opacity': .323 * u.m * u.m / u.kg},
        {'lambda': Q(.8, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 0,
         'opacity': .206 * u.m * u.m / u.kg},
        {'lambda': Q(.8, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 22,
         'opacity': .205 * u.m * u.m / u.kg},
        {'lambda': Q(.8, 'mm'),
         'size index': -3,
         'max size': Q(10, 'mm'),
         'porosity': 50,
         'ice fraction': 48,
         'opacity': .199 * u.m * u.m / u.kg}
    ]

    lambda_lo = Q(.45, 'mm')
    lambda_hi = Q(.8, 'mm')
    kappa_lo = {0: opacities[0]['opacity'],
                22: opacities[1]['opacity'],
                48: opacities[2]['opacity']}
    kappa_hi = {0: opacities[3]['opacity'],
                22: opacities[4]['opacity'],
                48: opacities[5]['opacity']}
    opacities.append({'lambda': wavelength,
                      'size index': -3,
                      'max size': Q(10, 'mm'),
                      'porosity': 50,
                      'ice fraction': 0,
                      'opacity': linterp(wavelength, lambda_lo, lambda_hi, kappa_lo[0], kappa_hi[0])})
    opacities.append({'lambda': wavelength,
                      'size index': -3,
                      'max size': Q(10, 'mm'),
                      'porosity': 50,
                      'ice fraction': 22,
                      'opacity': linterp(wavelength, lambda_lo, lambda_hi, kappa_lo[22], kappa_hi[22])})
    opacities.append({'lambda': wavelength,
                      'size index': -3,
                      'max size': Q(10, 'mm'),
                      'porosity': 50,
                      'ice fraction': 48,
                      'opacity': linterp(wavelength, lambda_lo, lambda_hi, kappa_lo[48], kappa_hi[48])})

    def mass(opacity):
        return neatm_A3.calcMassFlux(flux, opacity)
    masses = [dict(op, mass=mass(op['opacity'])) for op in opacities]
    print(f'{"lambda":>10} {"sz ind":>8} {"max sz":>8} {"porosity":>10} {"ice frac":>10} {"opacity":>14} {"mass":>12}')
    print(f'{"(mm)":>10} {"":>8} {"(mm)":>8} {"(%)":>10} {"(%)":>10} {"(m^2/kg)":>14} {"(kg)":>12}')
    for m in masses:
        print(f'{m["lambda"].m:>10.2} {m["size index"]:>8} {m["max size"].m:>8} {m["porosity"]:>10} {m["ice fraction"]:>10} {m["opacity"].value:>14.3} {m["mass"].value:>12.1e}')
# >>> my.run_Roth_NEATM_A3()
# Subsolar temperature = 606.0972431296984 K
# /home/zedd/.pyenv/versions/casaenv/lib/python3.10/site-packages/astropy/units/quantity.py:671: RuntimeWarning: overflow encountered in exp
#   result = super().__array_ufunc__(function, method, *arrays, **kwargs)
# bb, bb_err = 7.570260267633108, 1.2118511882874827e-07
# A3 point+coma fit diameter: 4.3+/-0.5 km
#     lambda   sz ind   max sz   porosity   ice frac        opacity         mass
#       (mm)              (mm)        (%)        (%)       (m^2/kg)         (kg)
#       0.45       -3       10         50          0          0.274      9.7e+07+/-     3.4e+07
#       0.45       -3       10         50         22            0.3      8.8e+07+/-     3.1e+07
#       0.45       -3       10         50         48          0.323      8.2e+07+/-     2.9e+07
#        0.8       -3       10         50          0          0.206      1.3e+08+/-     0.5e+08
#        0.8       -3       10         50         22          0.205      1.3e+08+/-     0.5e+08
#        0.8       -3       10         50         48          0.199      1.3e+08+/-     0.5e+08
#       0.65       -3       10         50          0          0.245      1.1e+08+/-     0.4e+08
#       0.65       -3       10         50         22           0.26      1.0e+08+/-     0.4e+08
#       0.65       -3       10         50         48          0.271      9.8e+07+/-     3.5e+07


def plot_Boissier_opacities():
    pass


def velocity_resolution(df, restfreq):
    return (df * my.c / restfreq).to('km/s')


def line_params(df, restfreq, start=Q(10, 'km/s')):
    # Make sure our increment is negative if we start at positive velocity
    if df > 0 and start > 0:
        df = -df
    dv = velocity_resolution(df, restfreq)
    nchan = round(2*start/abs(dv))
    return dv, start, int(nchan)


def make_cubes():
    vis = VIS_12M

    cell, imsize, _ = getcellsize(vis)
    cell = f'{cell}arcsec'

    def make_cube(name, df, restfreq, spw):
        print(f'cleaning {name}')

        dv, start, nchan = line_params(df, restfreq)
        print(f'dv = {dv:~P}')
        print(f'start = {start:~P}')
        print(f'nchan = {nchan}')

        imagedir = vis.parent / 'cubes' / name
        imagedir.mkdir(parents=True, exist_ok=True)

        imagename = imagedir / f'{name}.dirty'
        if not imagename.exists():
            casatasks.tclean(vis=str(vis), field='0', spw=spw, specmode='cubesource',
                             phasecenter='TRACKFIELD',
                             restfreq=f'{restfreq:~P}', width=f'{dv:~P}', start=f'{start:~P}',
                             nchan=nchan, restoringbeam='common', imagename=str(imagename),
                             cell=cell, imsize=imsize, outframe='REST', pbcor=True)

    make_cube('CH3OH_J7_K2_1_A',
              df=Q(488.281, 'kHz'),
              restfreq=Q(471.420467, 'GHz'),
              spw='3')
    make_cube('CH3OH_J9_8_K2_1_E',
              df=Q(-244.141, 'kHz'),
              restfreq=Q(460.875994, 'GHz'),
              spw='1')
    make_cube('CH3OH_J10_K2_1_A',
              df=Q(-244.141, 'kHz'),
              restfreq=Q(461.06417, 'GHz'),
              spw='1')
    make_cube('SO_N10_9_J11_10',
              df=Q(488.281, 'kHz'),
              restfreq=Q(471.5373786, 'GHz'),
              spw='3')


def clean_cubes():
    vis = VIS_12M

    cell, imsize, _ = getcellsize(vis)
    cell = f'{cell} arcsec'

    # Get threshold from spectral cube using above mask and averaging over channels that shouldn't contain the spectral line

    mask = vis.parent / 'comet.region'

    def clean_cube(name, df, restfreq, spw, threshold):
        print(f'cleaning {name}')

        dv, start, nchan = line_params(df, restfreq)
        print(f'dv = {dv:~P}')
        print(f'start = {start:~P}')
        print(f'nchan = {nchan}')

        imagedir = vis.parent / 'cubes' / name
        imagedir.mkdir(parents=True, exist_ok=True)

        imagename = imagedir / f'{name}.clean'
        if not imagename.exists():
            casatasks.tclean(vis=str(vis), field='0', spw=spw, specmode='cubesource',
                             phasecenter='TRACKFIELD', restfreq=f'{restfreq:~P}', width=f'{dv:~P}',
                             start=f'{start:~P}', nchan=nchan, restoringbeam='common',
                             imagename=str(imagename), cell=cell, imsize=imsize,
                             threshold=threshold, niter=50_000, mask=str(mask), outframe='REST',
                             pbcor=True)

    clean_cube('CH3OH_J7_K2_1_A',
               df=Q(488.281, 'kHz'),
               restfreq=Q(471.420467, 'GHz'),
               spw='3',
               threshold=f'{2*0.17427172}Jy')
    clean_cube('CH3OH_J9_8_K2_1_E',
               df=Q(-244.141, 'kHz'),
               restfreq=Q(460.875994, 'GHz'),
               spw='1',
               threshold=f'{2*0.00918207}Jy')
    clean_cube('CH3OH_J10_K2_1_A',
               df=Q(-244.141, 'kHz'),
               restfreq=Q(461.06417, 'GHz'),
               spw='1',
               threshold=f'{2*0.00918207}Jy')
    clean_cube('SO_N10_9_J11_10',
               df=Q(488.281, 'kHz'),
               restfreq=Q(471.5373786, 'GHz'),
               spw='3',
               threshold=f'{2*0.17427172}Jy')


# nucleus at coordinate 211, 76 in uncentered cont image
# nucleus at coordinate 120, 120 in centered cont image


def plot_spectra():
    methanol_J10 = np.loadtxt(
        '/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/spectrum.txt', skiprows=8).T
    methanol_J10_pbcor = np.loadtxt(
        '/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/spectrum.pbcor.txt', skiprows=8).T
    methanol_J9 = np.loadtxt(
        '/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/spectrum.txt', skiprows=8).T
    methanol_J9_pbcor = np.loadtxt(
        '/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/spectrum.pbcor.txt', skiprows=8).T
    methanol_J7 = np.loadtxt(
        '/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/spectrum.txt', skiprows=8).T
    methanol_J7_pbcor = np.loadtxt(
        '/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/spectrum.pbcor.txt', skiprows=8).T
    SO_N10 = np.loadtxt(
        '/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/spectrum.txt', skiprows=8).T
    SO_N10_pbcor = np.loadtxt(
        '/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/spectrum.pbcor.txt', skiprows=8).T

    _, ax = plt.subplots(1, 4, sharey=True)
    ax[0].plot(*methanol_J10, label='image')
    ax[0].plot(*methanol_J10_pbcor, label='pbcor')
    ax[0].legend()
    ax[0].set_title('CH3OH J10 K2->1 A+-')
    ax[1].plot(*methanol_J9, label='image')
    ax[1].plot(*methanol_J9_pbcor, label='pbcor')
    ax[1].legend()
    ax[1].set_title('CH3OH J9->8 K2->1 E')
    ax[2].plot(*methanol_J7, label='image')
    ax[2].plot(*methanol_J7_pbcor, label='pbcor')
    ax[2].legend()
    ax[2].set_title('CH3OH J7 K2->1 A+-')
    ax[3].plot(*SO_N10, label='image')
    ax[3].plot(*SO_N10_pbcor, label='pbcor')
    ax[3].legend()
    ax[3].set_title('SO N10->9 J11->10')
    ax[0].set_xlabel('radio velocity (km/s)')
    ax[0].set_ylabel('temperature (K)')
    plt.show()


@units.wraps(units.km / units.s, units.AU)
def line_width(heliocentric_distance):
    return 2 * Q(0.8 / math.sqrt(heliocentric_distance), 'km/s')


def line_width_A3():
    return line_width(Q(0.405766181926, 'AU'))


def line_widths_3sigma_upper_limit_A3():
    dv = line_width_A3()
    print(f'velocity range = {dv:~P}')

    def line_width_upper_limit(path, sigma):
        x, y = np.loadtxt(path, skiprows=8).T
        i = (x < -dv.magnitude) | (x > dv.magnitude)
        rms = Q(np.std(y[i]), 'K')
        nchan = x.shape[0] - np.count_nonzero(i)
        return sigma * rms * dv / math.sqrt(nchan / 2)

    spectra = {
        'CH3OH_J7_K2_1_A': '/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/spectrum.pbcor.txt',
        'CH3OH_J9_8_K2_1_E': '/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/spectrum.pbcor.txt',
        'CH3OH_J10_K2_1_A': '/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/spectrum.pbcor.txt',
        'SO_N10_9_J11_10': '/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/spectrum.pbcor.txt'
    }

    limits = {line: line_width_upper_limit(spectrum, 3) for line, spectrum in spectra.items()}

    return limits


def estimate_Q_H2O_A3():
    d = Q(0.74056486704488, 'AU').m
    r_minus_9 = Q(0.421744277212, 'AU').m
    m1 =  23.5 + 5 * math.log10(d) + 60 * math.log10(r_minus_9)
    Q_H2O = math.pow(10, 30.675 - 0.2453 * m1)
    print(f'Q_H2O = {Q_H2O:e}')
    # CASA <1>: my.estimate_Q_H2O_A3()
    # Q_H2O = 3.881117e+30


def find_parameter(f, y, lo=0, hi=1, tol=Q(.0005, 'K km/s')):
    '''Find x such that abs(f(x) - y) < tol for monotonically increasing function f.'''
    while True:
        err = f(lo) - y
        if err > 0:
            print(f'lo overshoots by {err}')
            # double interval by shifting lo downward
            lo = 2 * (hi - lo) - hi
        else:
            break
    while True:
        err = y - f(hi)
        if err > 0:
            print(f'hi undershoots by {err}')
            # double interval by shifting hi upward
            hi = 2 * (hi - lo) + lo
        else:
            break
    guess = (lo + hi) / 2
    iter = 1
    while True:
        print(f'iteration {iter}...')
        print(f'trying guess = {guess}...')
        err = f(guess) - y
        if abs(err) < tol:
            print(f'found parameter {guess} in {iter} iterations')
            return guess
        if err > 0:
            print(f'iteration {iter} overshot by {err}')
            hi = guess
            guess = (lo + hi) / 2
        else:
            print(f'iteration {iter} undershot by {err}')
            lo = guess
        guess = (lo + hi) / 2
        iter += 1


def run_sublime_all_lines_A3_slow():
    d = sublime2.Data()

    def initdata():
        d.Qratio = 1.0
        d.lp = 0
        d.dabund = 0.000
        d.openAngle = Q(90, 'degrees').m
        d.phase = Q(118.7183, 'degrees').m
        d.psAng = Q(264.727, 'degrees').m
        d.Q = Q(1.1e29, 's^-1').m
        d.rNuc = (Q(5.485749424508586, 'km').to('m') / 2).m
        d.rmax = 1e10
        d.betamain = 2.20318e-05
        d.reScaleLp = True
        d.rH = Q(0.405766181926, 'AU').m
        d.delta = Q(0.74056486704488, 'AU').m
        d.kernel = 'gauss'
        d.hpbwx = Q(0.48520079255104065, 'arcseconds').m
        d.hpbwy = Q(0.4330733120441437, 'arcseconds').m
        d.pa = Q(73.26548767089844, 'degrees').m
        d.eta = 0.7
        d.xne = 0.2
        d.lteonly = 0
        d.useEP = 0
        d.collPartId = 1
        d.colliScale = 1
        d.npix = 240
        d.imgres = Q(0.081, 'arcseconds').m
        d.pIntensity = 10000
        d.xoff = 0
        d.yoff = 0

        d.tkin = Q(100, 'K').m
        d.vexp = Q(0.8 / math.sqrt(d.rH), 'km/s').m
        d.vsource = Q(0, 'km/s').m
        d.doppler = Q(0, 'km/s').m
        d.fix = [True, True, True, True, True]

    limits = line_widths_3sigma_upper_limit_A3()
    pprint(limits, indent=4)
    

    def integrated_intensity(abund):
        d.abund = abund
        sublime2.run(d)
        return Q(d.Tdv, 'K km/s')

    abundances = {}

    # CH3OH_J7_K2_1_A
    initdata()
    df = Q(977, 'kHz')
    restfreq = Q(471.420467, 'GHz')
    dv, _, nchan = line_params(df, restfreq)
    d.rez = abs(dv).m
    d.chwid = (abs(dv) / 8).to('m/s').m
    d.nchan = nchan * 8
    d.betamol = 2.0764e-05
    d.spec = "CH3OH"
    d.trans = 288
    d.lamFile = '/home/zedd/alma/code/line_modeling/a-ch3oh_400K.dat'
    d.pumpFile = '/home/zedd/alma/code/line_modeling/g_a-ch3oh_400K_1au.dat'
    d.obsSpec = "/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/spectrum.pbcor.txt"
    d.popFile = '/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/pop.dat'
    d.modelFile = '/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/fit.c'
    d.fitsFile = '/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/CH3OH_288.fits'
    d.outPars = "/home/zedd/alma/12m/cubes/CH3OH_J7_K2_1_A/pars.out"

    print(f'finding abundance upper limit for CH3OH J=7 K 2->1 A+-...')
    abundances['CH3OH_J7_K2_1_A'] = 2 * find_parameter(integrated_intensity, limits['CH3OH_J7_K2_1_A'])

    # CH3OH_J9_8_K2_1_E
    initdata()
    df = Q(488, 'kHz')
    restfreq = Q(460.875994, 'GHz')
    dv, _, nchan = line_params(df, restfreq)
    d.rez = abs(dv).m
    d.chwid = (abs(dv) / 8).to('m/s').m
    d.nchan = nchan * 8
    d.betamol = 2.0764e-05
    d.spec = "CH3OH"
    d.trans = 523
    d.lamFile = '/home/zedd/alma/code/line_modeling/e-ch3oh_400K.dat'
    d.pumpFile = '/home/zedd/alma/code/line_modeling/g_e-ch3oh_400K_1au.dat'
    d.obsSpec = "/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/spectrum.pbcor.txt"
    d.popFile = "/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/pop.dat"
    d.modelFile = '/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/fit.c'
    d.fitsFile = '/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/CH3OH_523.fits'
    d.outPars = "/home/zedd/alma/12m/cubes/CH3OH_J9_8_K2_1_E/pars.out"

    print(f'finding abundance upper limit for CH3OH J 9->8 K 2->1 E...')
    abundances['CH3OH_J9_8_K2_1_E'] = 2 * find_parameter(integrated_intensity, limits['CH3OH_J9_8_K2_1_E'])

    # CH3OH_J10_K2_1_A
    initdata()
    df = Q(488, 'kHz')
    restfreq = Q(461.06417, 'GHz')
    dv, _, nchan = line_params(df, restfreq)
    d.rez = abs(dv).m
    d.chwid = (abs(dv) / 8).to('m/s').m
    d.nchan = nchan * 8
    d.betamol = 2.0764e-05
    d.spec = "CH3OH"
    d.trans = 283
    d.lamFile = '/home/zedd/alma/code/line_modeling/a-ch3oh_400K.dat'
    d.pumpFile = '/home/zedd/alma/code/line_modeling/g_a-ch3oh_400K_1au.dat'
    d.obsSpec = "/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/spectrum.pbcor.txt"
    d.popFile = "/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/pop.dat"
    d.modelFile = '/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/fit.c'
    d.fitsFile = '/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/CH3OH_523.fits'
    d.outPars = "/home/zedd/alma/12m/cubes/CH3OH_J10_K2_1_A/pars.out"

    print(f'finding abundance upper limit for CH3OH J=10 K 2->1 A+-...')
    abundances['CH3OH_J10_K2_1_A'] = 2 * find_parameter(integrated_intensity, limits['CH3OH_J10_K2_1_A'])

    # SO_N10_9_J11_10
    initdata()
    df = Q(977, 'kHz')
    restfreq = Q(471.5373786, 'GHz')
    dv, _, nchan = line_params(df, restfreq)
    d.rez = abs(dv).m
    d.chwid = (abs(dv) / 8).to('m/s').m
    d.nchan = nchan * 8
    d.betamol = 6.6399e-04
    d.spec = 'SO'
    d.trans = 107
    d.lamFile = '/home/zedd/alma/code/line_modeling/SO-out.dat'
    d.pumpFile = ''
    d.obsSpec = "/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/spectrum.pbcor.txt"
    d.popFile = "/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/pop.dat"
    d.modelFile = '/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/fit.c'
    d.fitsFile = '/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/SO_106.fits'
    d.outPars = "/home/zedd/alma/12m/cubes/SO_N10_9_J11_10/pars.out"

    print(f'finding abundance upper limit for SO N 10->9 J 11->10...')
    abundances['SO_N10_9_J11_10'] = find_parameter(integrated_intensity, limits['SO_N10_9_J11_10'])

    # https://home.strw.leidenuniv.nl/~moldata/NH3.html
    # NH3
    # NH2D
    # multiply abundances by 2 to account for A+- and E pops

    return abundances


# Abundance upper limits
# {'CH3OH_J7_K2_1_A': 0.03757476806640625,
#  'CH3OH_J9_8_K2_1_E': 0.014129638671875,
#  'CH3OH_J10_K2_1_A': 0.0173187255859375,
#  'SO_N10_9_J11_10': 0.00528717041015625}


if __name__ == '__main__':
    clean_cubes()
