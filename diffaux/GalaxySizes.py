import numpy as np
import os
import sys
import matplotlib
#matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rc('text.latex', preamble=r'\usepackage{bm}')
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
from os.path import expanduser
from itertools import zip_longest
home = expanduser("~")
host = os.getenv('HOSTNAME')
print(host)


# In[3]:


if host is not None:
    if 'bebop' in host:
        input_dir = '/lcrc/project/galsampler/Catalog_5000/OR_5000/'
        size_dir = '/lcrc/project/galsampler/Validation_Data/Sizes'
else:
    #input_dir = '/lus/eagle/projects/LastJourney/kovacs/Catalog_5000/OR_5000/'
    #size_dir = '/lus/eagle/projects/LastJourney/kovacs/Validation_Data/Sizes'
    #laptop
    input_dir = '/Users/kovacs/Catalog_5000/OR_5000/roman_rubin_2023_v1.1.3'
    size_dir = '/Users/kovacs/Validation_Data/Sizes'
    notebook_dir = '/Users/kovacs/cosmology/diff_notebooks'
fname = 'roman_rubin_2023_z_*_cutout_{}.hdf5'
from diffaux.validation.get_catalog_data import get_fhlist
from diffaux.validation.get_catalog_data import get_colnames
#from lsstdesc_diffsky.write_mock_to_disk import get_astropy_table
from diffaux.size_modeling import zhang_yang17


# In[150]:




# In[187]:



# In[152]:



authors = get_author_list(validation_info['Re_vs_z'])


# In[177]:


Rmdata = {}
authors_rm = get_author_list(validation_info['Re_vs_Mstar'])
Rmdata= read_size_data(Rmdata, authors, info_key='Re_vs_Mstar')
#print(Rmdata['Re_vs_Mstar']['mowla_2019'])


# In[155]:


#print(data['Re_vs_z']['george_2024_3000'].keys())
Redata = {}
authors_re = get_author_list(validation_info['Re_vs_z_data'])
Redata = read_size_data(Redata, authors, info_key='Re_vs_z_data')


# In[156]:


#for sample in validation_info['Re_vs_Mstar']['samples']:
#    data= read_size_data(data, authors, info_key='Re_vs_Mstar', sample=sample)
data = {}
data= read_size_data(data, authors, info_key='Re_vs_z')


# In[185]:



make_data_plots(data, validation_info, authors, info_keys=['Re_vs_z'], summary_only=True)


# In[188]:


make_data_plots(Rmdata, validation_info, authors_rm, info_keys=['Re_vs_Mstar'], summary_only=False)


# In[ ]:


# fit data points for B and beta with a power law in M*
from scipy.optimize import curve_fit
    
# collect data points
def get_data_vectors(data, val_info, sample='Star-forming', lambda_min=0.5, lambda_max=1.0,
                     X='x-values', Y='y-values', dYp='y-errors+', dYn='y-errors-'):
    #initialize
    xvec = np.asarray([])
    yvec = []
    dyvec = []
    for yname in val_info[Y]:
        yvec.append(np.asarray([]))
        dyvec.append(np.asarray([]))
    for k, v in data.items():
        wave = val_info[k]['wavelength']
        if wave >= lambda_min and wave <= lambda_max:
            print('Processing {} {}'.format(k, wave))
            add_xvec = True
            #print(val_info[X], xvec)
            for n, (y, dy, yn, dynp, dynn) in enumerate(zip(yvec, dyvec, val_info[Y],
                                                            val_info[dYp], val_info[dYn])):
                #print(yname,  v[yname.format(sample)])
                mask = np.abs(v[yn.format(sample)]) > 0.
                if add_xvec:
                    xvec = np.concatenate((xvec, v[val_info[X]][mask]))
                    add_xvec = False
                y = np.concatenate((y, v[yn.format(sample)][mask]))
                yerr = np.fmax(v[dynp.format(sample)], v[dynn.format(sample)])
                dy = np.concatenate((dy, yerr[mask]))
                yvec[n] = y
                dyvec[n] = dy
        else:
            print('Skipping {} {}'.format(k, wave))
            
        

    assert(all([len(xvec) == len(y) for y in yvec])), 'Mismatch in assembled data vectors'
    return xvec, yvec, dyvec


# In[ ]:


data_vectors = {}
fit_type = 'Re_vs_z'
data_vectors['Re_vs_z'] = {}
samples = ['Star-forming', 'Quiescent']
for sample in samples:
    data_vectors[fit_type][sample] = {}
    x, y, dy = get_data_vectors(data[fit_type], validation_info[fit_type], sample=sample)
    data_vectors[fit_type][sample][validation_info[fit_type]['x-values']] = x
    data_vectors[fit_type][sample]['y'] = y
    data_vectors[fit_type][sample]['dy'] = dy

author_list = get_author_list(validation_info['Re_vs_z'], lambda_min=0.5, lambda_max=1.0)


# In[ ]:


def pwrfit(x, x0, a, b):
    return a*np.power(x-x0, b)

def expfit(x, x0, a, b):
    return a*np.exp(b*(x-x0))

def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (x - x0)))

#The variable xtp is not a free parameter, it is simply the abscissa value at which the normalization free parameter ytp is defined
def _sig_slope(x, xtp, ytp, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return ytp + slope * (x - xtp)

def fit_parameters(data_vectors, samples, val_info, p0_values,
                   func=_sigmoid):
    parameters = val_info['y-values']
    fits = {}
    for sample, p0_row in zip(samples, p0_values):
        M = data_vectors[sample][val_info['x-values']]
        for par, p0, y, dy in zip(parameters, p0_row,
                                  data_vectors[sample]['y'],
                                  data_vectors[sample]['dy']):
            #print(par, M, y, dy, p0)
            fit_key = par.format(sample)
            fits[fit_key] = {}
            popt, pcov = curve_fit(func, M, y, sigma=dy, p0=p0, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            fits[fit_key]['popt'] = popt
            fits[fit_key]['perr'] = perr
            fits[fit_key]['pcov'] = pcov
            print('Fit parameters: ', fit_key, popt)
            print('Errors: ', fit_key, perr)

    #print(fits)
    return fits

def plot_fits(fits, data_vectors, samples, val_info, data={}, authors=None,
              func=_sigmoid, title='', dYp='y-errors+', dYn='y-errors-',
              plotdir=os.path.join(notebook_dir, 'SizePlots'), fontsize=12,
              pltname='Fit_Re_vs_z{}.png', fit_label='\\log(M^*/M_\\odot)',
             ):
    
    parameters = val_info['y-values']
    nrow = len(samples)
    ncol = len(parameters)
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
    colors = ('blue', 'red') #for simple plot
    for sample, ax_row, color in zip(samples, ax_all, colors):
        M = data_vectors[sample][val_info['x-values']]
        for par, ax, y, dy, dYp, dYn, xlabel, ylabel in zip(parameters, ax_row,
                                                            data_vectors[sample]['y'],
                                                            data_vectors[sample]['dy'],
                                                            val_info['y-errors+'],
                                                            val_info['y-errors-'],
                                                            val_info['xlabels'],
                                                            val_info['ylabels'],
                                                            ):
            fit_key = par.format(sample)
            if not authors:
                ax.errorbar(M, y, yerr=dy, label=fit_key, color=color, marker='o', linestyle='')
            else: #fancy plot with full legend
                for author in authors:
                    idx = val_info[author]['samples'].index(sample)
                    mcolor = val_info[author]['colors'][idx]
                    xcol = val_info['x-values']
                    ycol = fit_key
                    yerr = np.fmax(data[author][dYp.format(sample)],
                                   data[author][dYn.format(sample)])
                    #mask for missing values
                    yvalues = data[author][fit_key]
                    mask = np.abs(yvalues) > 0.
                    marker = val_info[author]['marker']
                    sum_label = '{} ({}) ${}\\mu m$'.format(sample, val_info[author]['short_title'],
                                                val_info[author]['wavelength'])
                    ax.errorbar(data[author][xcol][mask], yvalues[mask],
                                yerr=yerr[mask], label=sum_label,
                                color=mcolor, marker=marker, linestyle='')
            popt =  fits[fit_key]['popt']
            Msort = np.sort(M)
            label = '$\\bm{{{:.2g}+({:.2g}-{:.2g})/(1+\\exp(-{:.2g}*({} -{:.3g})))}}$'.format(
                    popt[2], popt[3], popt[2], popt[1], fit_label, popt[0])
            ax.plot(Msort, func(Msort, *popt), color='black', label=label)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_title(sample)
            legend_properties = {'weight':'bold'}
            ax.legend(loc='best', fontsize='small', prop=legend_properties)

    #fig.suptitle('{} ({})'.format())
    save_fig(fig, plotdir, pltname.format(title))        
    return


# In[ ]:


p0_values = [[(11.5, 2.7, 3.0, 40.0), (11.3, 2.5, 0.15, 2.5)], [(11.0, 3.6, 1.5, 16.0), (10.0, 8.0, 0.4, 1.2)]]
fit_type='Re_vs_z'
# test p0 values initialize fits with p0 values and plot
fits={}
fits[fit_type] = {}
for sample, p0_row in zip(samples, p0_values):
    for par, p0 in zip(validation_info[fit_type]['y-values'], p0_row):
        fit_key = par.format(sample)
        fits[fit_type][fit_key] = {} #if fit_key not in fits[fit_type].keys()
        fits[fit_type][fit_key]['popt'] = p0

plot_fits(fits[fit_type], data_vectors[fit_type], samples, validation_info[fit_type], func=_sigmoid, title='_InitialGuess')
fits[fit_type] = fit_parameters(data_vectors[fit_type], samples, validation_info[fit_type], p0_values, func=_sigmoid)
plot_fits(fits[fit_type], data_vectors[fit_type], samples, validation_info[fit_type], func=_sigmoid,
          data=data[fit_type], authors=author_list)


# In[ ]:


def median_size_vs_z(z, B, beta):
    Re_med = B*np.power(1+z, -beta)
    return Re_med

def get_color_mask(color, sample, UVJcolor_cut=1.5, UVJ=True):
    mask = np.ones(len(color), dtype=bool)
    if sample == 'Star-forming':
        if UVJ:
            mask = (color < UVJcolor_cut)
        else:
            print('Unknown color option')
    else:
        if UVJ:
            mask = (color >= UVJcolor_cut)
        else:
            print('Unknown color option')    
    return mask

def get_median_sizes(fits, log_Mstar, redshift, color, Ngals, samples, parameters,
                     UVJcolor_cut=1.5, fit_func=_sigmoid, size_func=median_size_vs_z):
    R_med = np.zeros(Ngals) 
    #determine parameter values from fits 
    for sample in samples:
        mask = get_color_mask(color, sample, UVJcolor_cut=UVJcolor_cut)
        #print(sample, np.count_nonzero(mask), mask)
        fit_parameters = []
        for par in parameters:
            parameter = par.format(sample)
            popt = fits[parameter]['popt']
            #print(parameter, popt)
            fit_parameters.append(fit_func(log_Mstar[mask], *popt))
 
        R_med[mask] = size_func(redshift[mask], *fit_parameters)

    return R_med

def get_scatter(R_med, scatter_hi, scatter_lo):
    scatter_up = R_med*(np.power(10, scatter_hi) - 1)
    scatter_down = R_med*(1 - np.power(10, -scatter_lo))
    return scatter_up, scatter_down

def generate_sizes(fits, log_Mstar, redshift, color, parameters,
                   samples=['Star-forming', 'Quiescent'],
                   UVJcolor_cut=1.5, scatter_hi=0.2, scatter_lo=0.2,
                   fit_func=_sigmoid, size_func=median_size_vs_z,
                   ):
    """
    fits: dictionary of fit parameters
    log_Mstar: array length (Ngals), log10(stellar masses) of galaxies in units of Msun
    redshift: array length Ngals, redshift of galaxies
    color: array length Ngals, color of galaxies

    returns
    sizes: array length (Ngals), size in kpc

    """
    Ngals = len(log_Mstar)
    assert(len(redshift)==Ngals), "Supplied redshifts don't match length of M* array" 
    assert(len(color)==Ngals), "Supplied colors don't match length of M* array"

    R_med = get_median_sizes(fits, log_Mstar, redshift, color, Ngals, samples, parameters,
                             UVJcolor_cut=UVJcolor_cut,
                             fit_func=fit_func, size_func=size_func)
    scatter_up, scatter_down  = get_scatter(R_med, scatter_hi=scatter_hi, scatter_lo=scatter_lo)

    sizes_hi = np.random.normal(loc=R_med, scale=scatter_hi, size=Ngals)
    sizes_lo = np.random.normal(loc=R_med, scale=scatter_lo, size=Ngals)       

    return np.where(sizes_lo < R_med, sizes_lo, sizes_hi), R_med, scatter_up, scatter_down 
    


# In[ ]:


#test sizes
N = 5000
lM_lo = 9.0
lM_hi = 12.0
z_lo = 0.
z_hi = 3.0
log_Mstar = np.random.uniform(low=lM_lo, high=lM_hi, size=N)
redshift = np.random.uniform(low=z_lo, high=z_hi, size=N)
color_gal = np.random.uniform(low=-0.2, high=2.3, size=N)
#print(log_Mstar, redshift, color)
Re, R_med, scatter_up, scatter_down = generate_sizes(fits[fit_type], log_Mstar, redshift, color_gal,
                                                                         validation_info[fit_type]['y-values'])
print(np.min(R_med), np.max(R_med))
print(np.min(Re), np.max(Re))


# In[ ]:


from scipy.stats import binned_statistic
qs = [Re, R_med, scatter_up, scatter_down]
nrow, ncol = get_nrow_ncol(len(qs))
fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
for ax, q in zip(ax_all.flat, qs):
    ax.hist(q, bins=40)


# In[ ]:


tests = [{}, {}]
print(samples)
for test, ikey in zip(tests, ['Re_vs_z_data', 'Re_vs_Mstar_data']):
    if 'Mstar' in ikey:
        for sample in samples:
            test= read_size_data(test, authors, info_key=ikey, sample=sample)
    else:
        test= read_size_data(test, authors, info_key=ikey)
    print(test[ikey].keys())


# In[ ]:




# In[ ]:


# make validation plots
for test, ikey in zip(tests, ['Re_vs_z_data', 'Re_vs_Mstar_data']):
    authors = get_author_list(validation_info[ikey])
    plot_generated_sizes(Re, R_med, color_gal, log_Mstar, redshift, samples, 
                         authors, test[ikey], validation_info[ikey])


# In[ ]:




