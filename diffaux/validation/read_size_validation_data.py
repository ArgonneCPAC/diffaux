"""
Module implements reading of validation data to be used for size model development and validation;
validation_info dictionary supplies information for reading and plotting validation data.

"""

import numpy as np
import os
from importlib.resources import files
from .plot_size_data import plot_Re_vs_z_Mstar_data, plot_Re_vs_z_Mstar_fits

SIZE_DATA_DIR = files('diffaux').joinpath('data/GalaxySizes')


def get_author_list(val_info, lambda_min=0.0, lambda_max=2.0):
    authors = []
    for k, v in val_info.items():
        # fix this line if other keys added to validation_info
        if 'x-' in k or 'y-' in k or 'z-' in k or 'read' in k or 'label' in k or '_values' in k or 'titles' in k or 'plot' in k:
            continue
        if v['wavelength'] >= lambda_min and v['wavelength'] <= lambda_max:
            authors.append(k)
        else:
            print(
                'Skipping {} {} wavelength data'.format(
                    k, val_info[k]['wavelength']))

    print('authors = ', authors)

    return authors


def read_Re_vs_Mstar_data(tmp, data, author, validation_info, sample=''):
    # collect M* values if not already in dict
    v = validation_info[author]
    if not all([cname in data.keys() for cname in v['M*_colnames']]):
        for col, colname in zip(v['M*_columns'], v['M*_colnames']):
            data[colname] = tmp[:, col]

    # collect M_med, R_e values for each z bin
    for n, zlo in enumerate(np.arange(v['z_lo'], v['z_hi'], v['dz'])):
        zhi = zlo + v['dz']
        # print(n, zlo, zhi)
        zlabel = '${:.1f} \\leq z < {:.1f}$'.format(zlo, zhi)
        print('Processing {}'.format(zlabel))
        if zlabel not in data.keys():
            data[zlabel] = {}
        for i, colname in enumerate(v['data_colnames']):
            col = len(v['M*_columns']) + i + n * len(v['data_colnames'])
            data[zlabel][colname.format(sample)] = tmp[:, col]

        data[zlabel] = complete_data_values(
            data[zlabel], author, validation_info, samples=sample)

    return data


def read_Re_vs_z_data(tmp, data, author, validation_info):

    # collect M* columns
    v = validation_info[author]
    if 'format' in v.keys() and 'M*_by_row' in v['format']:
        row = v['M*_rows']
        data[v['M*_rownames']] = tmp[row][tmp[row]
                                          > validation_info['filling_values']]
        # save colnames for later concatenation
        M_old = v['M*_rownames']
        M_new = validation_info['x-values']
        data[M_new] = np.asarray([])
        Mstars = np.unique(data[M_old])
        length = len(np.unique(data[M_old]))
    else:
        for col, colname in zip(v['M*_columns'], v['M*_colnames']):
            data[colname] = tmp[:, col]
    # collect z columns
    for col, z_name in zip(v['z_columns'], v['z_colnames']):
        if 'format' in v.keys() and 'M*_by_row' in v['format']:
            # save colnames for later concatenation (since only one z column in
            # data)
            data[z_name] = tmp[v['M*_rows'] + 1:, col]
            z_new = validation_info['z-values']
            data[z_new] = np.asarray([])
        else:
            data[z_name] = tmp[:, col]

    # collect size data
    for n, sample in enumerate(validation_info[author]['samples']):
        for i, colname in enumerate(v['data_colnames']):
            if 'format' in v.keys() and 'M*_by_row' in v['format']:
                # duplicate M* rows
                for j, Mstar in enumerate(Mstars):
                    col = len(v['z_columns']) + i + j * len(v['data_colnames']
                                                            ) + n * length * len(v['data_colnames'])
                    colname = colname.format(sample)
                    if colname not in data.keys():
                        data[colname] = np.asarray([])
                    # concatenate data columns
                    data[colname] = np.concatenate(
                        (data[colname], tmp[v['M*_rows'] + 1:, col]))
                    # concatenate z amd M* arrays
                    if i == 0 and n == 0:
                        data[z_new] = np.concatenate(
                            (data[z_new], data[z_name]))
                        M_repeat = np.array([Mstar] * len(data[z_name]))
                        data[M_new] = np.concatenate((data[M_new], M_repeat))
            else:
                col = len(v['M*_columns']) + len(v['z_columns']) + \
                    i + n * len(v['data_colnames'])
                colname = colname.format(sample)
                data[colname] = tmp[:, col]

    data = complete_data_values(data, author, validation_info)

    # check
    for k, v in data.items():
        if '_row' not in k:
            test = len(data[validation_info['z-values']])
            assert (
                len(v) == test), 'Mismatch in length of array {}, {} != {}'.format(
                k, len(v), test)

    return data


def read_Re_vs_z_Mstar_fits(tmp, data, author, validation_info):

    # collect M* columns
    v = validation_info[author]
    for col, colname in zip(v['x_columns'], v['x_colnames']):
        data[colname] = tmp[:, col]

    # collect info on fit parameters
    for n, sample in enumerate(validation_info[author]['samples']):
        for i, colname in enumerate(v['data_colnames']):
            col = len(v['x_columns']) + i + n * len(v['data_colnames'])
            colname = colname.format(sample)
            data[colname] = tmp[:, col]
            # assume only one sign flip required
            if 'flip_sign' in v.keys(
            ) and colname == v['data_colnames'][v['flip_value_columns']].format(sample):
                data[colname] = -data[colname]
                print('Flipping sign of {}'.format(colname))

    data = complete_data_values(data, author, validation_info)

    return data


def complete_data_values(data, author, validation_info, samples=''):

    # Check for logarithmic values in data and single-sided error values
    # and complete columns so that all data sets match
    v = validation_info[author]
    if not samples:
        samples = v['samples']
    if type(samples) is str:
        samples = [samples]
    for sample in samples:
        xval = validation_info['x-values'].format(sample)
        if xval not in data.keys() and 'copy_from_common' in v.keys(
        ) and v['copy_from_common']:
            old_x = v['copy_columns']
            data[xval] = data[old_x]
            print('..Copying {} to {}'.format(old_x, xval))

        for n, (yval, yerr_p, yerr_n) in enumerate(zip(validation_info['y-values'],
                                                       validation_info['y-errors+'],
                                                       validation_info['y-errors-'])):
            yval = yval.format(sample)
            yerr_p = yerr_p.format(sample)
            yerr_n = yerr_n.format(sample)
            if yval not in data.keys():
                # convert value columns and error columns to or from log
                val_idx = v['value_columns'][n]
                err_idx = v['error_columns'][n]
                old_val = v['data_colnames'][val_idx].format(sample)
                data[yval] = np.zeros(len(data[old_val]))
                old_err = v['data_colnames'][err_idx].format(sample)
                data[yerr_p] = np.zeros(len(data[old_val]))
                if 'error_n_columns' in v.keys():
                    err_n_idx = v['error_n_columns'][n]
                    data[yerr_n] = np.zeros(len(data[old_val]))
                    old_err_n = v['data_colnames'][err_n_idx].format(sample)
                mask = data[old_val] != 0
                if 'convert_to_log' in v.keys() and v['convert_to_log']:
                    data[yval][mask] = np.log10(data[old_val][mask])
                    print('..Converting {} to log'.format(old_val))
                    data[yerr_p][mask] = np.log10(data[old_val][mask]
                                                  + data[old_err][mask]) - data[yval][mask]
                    print('..Converting {} to log'.format(old_err))
                elif 'convert_from_log' in v.keys() and v['convert_from_log']:
                    # conversion of log(beta) to beta; (beta should be +ve)
                    data[yval][mask] = np.power(10, data[old_val][mask])
                    # print(data[old_val][mask], '->', data[yval][mask])
                    print(
                        '..Converting {} from log to {}'.format(
                            old_val, yval))
                    if 'convert_limits_to_errors' in v.keys(
                    ) and v['convert_limits_to_errors']:
                        # print('Need to check convert limits')
                        data[yerr_p][mask] = np.abs(
                            np.power(10, data[old_err][mask]) - data[yval][mask])
                        # Check for asymmetric errors
                        if 'error_n_columns' in v.keys():
                            data[yerr_n][mask] = np.abs(
                                -np.power(10, data[old_err_n][mask]) + data[yval][mask])
                    else:
                        data[yerr_p][mask] = np.abs(np.power(10, (data[old_val][mask]
                                                    + data[old_err][mask])) - data[yval][mask])
                        # Check for asymmetric errors
                        if 'error_n_columns' in v.keys():
                            data[yerr_n][mask] = np.abs(-np.power(10, (data[old_val][mask]
                                                                       - data[old_err_n][mask])) + data[yval][mask])
                    print(
                        '..Converting {} from log to {}'.format(
                            old_err, yerr_p))

                # else Nothing to do
            # Check for negative errors
            if yerr_n not in data.keys():
                data[yerr_n] = data[yerr_p]
                print('..Assigning symmetric errors for {}'.format(yerr_n))

         # adjustment for pivot point for M* fits
        if 'adjust_for_M_p' in v.keys() and v['adjust_for_M_p']:
            Mp_txt = v['M_p_label']
            yval = validation_info['y-values'][v['M_p_value_column']
                                               ].format(sample)
            yerr_p = validation_info['y-errors+'][v['M_p_value_column']
                                                  ].format(sample)
            yerr_n = validation_info['y-errors-'][v['M_p_value_column']
                                                  ].format(sample)
            alpha = data[validation_info['y-values']
                         [v['M_p_power_column']].format(sample)]
            M_p_rescale = np.power(v['M_p_factor'], alpha)
            for q in [yval, yerr_p, yerr_n]:
                data[q + Mp_txt] = data[q] / M_p_rescale
                print('..Dividing {} by M_p conversion factor {:.1f}**alpha -> {}{}'.format(q, v['M_p_factor'],
                                                                                            q, Mp_txt))

    return data


"""
Dictionary to supply information for reading and plotting validation data for size modeling
"""

validation_info = {'Re_vs_Mstar_data': {'missing_values': '--',
                                        'filling_values': 0.0,
                                        'reader': read_Re_vs_Mstar_data,
                                        'plotter': plot_Re_vs_z_Mstar_data,
                                        'xlabel': '$\\bm{\\log_{10}(M^*/M_\\odot)}$',
                                        'ylabel': '$\\bm{R_{e} (\\mathrm{kpc})}$',
                                        'lgnd_label': '$\\bm{R_e}$',
                                        'x-values': 'M*_med_{}',
                                        'y-values': ['Re_{}'],
                                        'y-errors+': ['dRe_{}+'],
                                        'y-errors-': ['dRe_{}-'],
                                        'martorano_2024': {
                                            'samples': ['All', 'Starforming', 'Quiescent'],
                                            'colors': ['black', 'darkslateblue', 'firebrick'],
                                            'filename': 'martorano_2024_table1_Mstar_log10Re_{}.txt',
                                            'skip_header': 10,
                                            'dz': .5,
                                            'z_lo': 0.5,
                                            'z_hi': 2.5,
                                            'M*_columns': [0, 1],
                                            'M*_colnames': ['M*_lo', 'M*_hi'],
                                            'data_colnames': ['M*_med_{}', 'log10_Re_16%_{}', 'log10_Re_50%_{}', 'log10_Re_84%_{}'],
                                            'value_columns': [2],
                                            'error_columns': [3],
                                            'error_n_columns': [1],
                                            'convert_from_log': True,
                                            'convert_limits_to_errors': True,
                                            'wavelength': 1.5,
                                            'marker': 's',
                                            'short_title': 'JWST+COSMOS',
                                            'suptitle': 'Martorano et. al. 2024, JWST+COSMOS',
                                        },
                                        },
                   'Re_vs_Mstar': {'missing_values': '--',
                                   'filling_values': 0.0,
                                   'reader': read_Re_vs_z_Mstar_fits,
                                   'plotter': plot_Re_vs_z_Mstar_fits,
                                   'y-values': ['A_{}', 'alpha_{}'],
                                   'y-errors+': ['dA_{}+', 'dalpha_{}+'],
                                   'y-errors-': ['dA_{}-', 'dalpha_{}-'],
                                   'xlabels': ['$\\bm{z}$',
                                               '$\\bm{z}$'],
                                   'ylabels': ['$\\bm{A}$ (kpc)', r'$\bm\alpha$'],
                                   'titles': ['$\\bm{{A}}$: $\\bm{{R_e = A*(M^*/({} \\times 10^{{10}}M_\\odot))^{{\\alpha}}}}$',
                                              '$\\bm{{\\alpha}}$: $\\bm{{R_e = A*(M^*/({} \\times 10^{{10}}M_\\odot))^{{\\alpha}}}}$'],
                                   'x-values': 'z_med_{}',
                                   'kawinwanichakij_2021': {
                                       'samples': ['Quiescent', 'Starforming'],
                                       'colors': ['firebrick', 'royalblue'],
                                       'filename': 'kawinwanichakij_Table5b_Re_vs_Mstar_fits.txt',
                                       'skip_header': 4,
                                       'M_p': 5,
                                       'x_colnames': ['z_med'],
                                       'x_columns': [0],
                                       'data_colnames': ['alpha_{}', 'dalpha_{}+', 'A_{}', 'dA_{}+',
                                                         'sigma_{}', 'dsigma_{}+'],
                                       'copy_from_common': True,
                                       'copy_columns': 'z_med',
                                       'marker': 'D',
                                                 'wavelength': 0.5,
                                                 'short_title': 'HSC',
                                                 'suptitle': 'Kawinwanichakij et. al. 2021, HSC',
                                   },
                                   'mowla_2019': {
                                       'samples': ['All', 'Starforming', 'Quiescent'],
                                       'colors': ['black', 'mediumblue', 'crimson'],
                                       'filename': 'mowla_2019_Table2_Re_vs_Mstar_fits.txt',
                                       'skip_header': 4,
                                       'M_p': 7,
                                       'x_colnames': ['z_med'],
                                       'x_columns': [0],
                                       'data_colnames': ['logA_{}', 'dlogA_{}+',
                                                         'alpha_{}', 'dalpha_{}+'],
                                       'value_columns': [0],
                                       'error_columns': [1],
                                       'convert_from_log': True,
                                       'adjust_for_M_p': True,
                                       'M_p_factor': 1.4,
                                       'M_p_value_column': 0,
                                       'M_p_power_column': 1,
                                       'M_p_label': '_M_p_rescaled',
                                       'copy_from_common': True,
                                       'copy_columns': 'z_med',
                                       'marker': 'o',
                                                 'wavelength': 0.55,
                                                 'short_title': 'COSMOS-DASH',
                                                 'suptitle': 'Mowla et al. 2019, COSMOS-DASH',
                                   },
                                   'george_2024_3000': {
                                       'samples': ['Starforming', 'Quiescent'],
                                       'colors': ['mediumslateblue', 'crimson'],
                                       'filename': 'George_2024_Table2_3000_Re_vs_Mstar_fits.txt',
                                       'skip_header': 4,
                                       'M_p': 5,
                                       'x_colnames': ['zlo', 'zhi'],
                                       'x_columns': [0, 1],
                                       'data_colnames': ['z_med_{}', 'dz_med_{}', 'alpha_{}',
                                                         'dalpha_{}+', 'A_{}', 'dA_{}+',
                                                         'sigma_{}', 'dsigma_{}+',
                                                         ],
                                       'marker': 'X',
                                       'wavelength': 0.3,
                                       'short_title': 'CLAUDS+HSC',
                                       'suptitle': 'George et. al. 2024, CLAUDS+HSC',
                                   },
                                   'george_2024_5000': {
                                       'samples': ['Starforming', 'Quiescent'],
                                       'colors': ['mediumslateblue', 'crimson'],
                                       'filename': 'George_2024_Table2_5000_Re_vs_Mstar_fits.txt',
                                       'skip_header': 4,
                                       'M_p': 5,
                                       'x_colnames': ['zlo', 'zhi'],
                                       'x_columns': [0, 1],
                                       'data_colnames': ['z_med_{}', 'dz_med_{}', 'alpha_{}',
                                                         'dalpha_{}+', 'A_{}', 'dA_{}+',
                                                         'sigma_{}', 'dsigma_{}+',
                                                         ],
                                       'marker': 'X',
                                       'wavelength': 0.5,
                                       'short_title': 'CLAUDS+HSC',
                                       'suptitle': 'George et. al. 2024, CLAUDS+HSC',
                                   },

                                   },
                   'Re_vs_z_data': {'missing_values': '--',
                                    'filling_values': 0.0,
                                    'reader': read_Re_vs_z_data,
                                    'plotter': plot_Re_vs_z_Mstar_data,
                                    'xlabel': '$\\bm{\\log_{10}(M^*/M_\\odot)}$',
                                    'ylabel': '$\\bm{R_{e}^{med} (\\mathrm{kpc})}$',
                                    'lgnd_label': '$\\bm{R_e^{med}}$',
                                    'x-values': 'M*_med',
                                    'y-values': ['Re_med_{}'],
                                    'z-values': 'z_med',
                                    'y-errors+': ['dRe_med_{}+'],
                                    'y-errors-': ['dRe_med_{}-'],
                                    'mowla_2019': {
                                        'samples': ['All', 'Starforming', 'Quiescent'],
                                        'colors': ['black', 'mediumblue', 'crimson'],
                                        'filename': 'mowla_2019_Table3_Re_vs_z_data.txt',
                                        'skip_header': 4,
                                        'format': 'M*_by_row',
                                        'M*_rownames': 'M*_med_row',
                                        'M*_rows': 0,
                                        'z_colnames': ['z_med_row'],
                                        'z_columns': [0],
                                        'data_colnames': ['Re_med_{}', 'dRe_med_{}+'],
                                        'marker': 'o',
                                        'wavelength': 0.55,
                                        'short_title': 'COSMOS-DASH',
                                        'suptitle': 'Mowla et al. 2019, COSMOS-DASH',
                                    },
                                    'kawinwanichakij_2021': {
                                        'samples': ['All', 'Quiescent', 'Starforming'],
                                        'colors': ['black', 'firebrick', 'royalblue'],
                                        'filename': 'kawinwanichakij_Table6a_Re_median_vs_z.txt',
                                        'skip_header': 2,
                                        'format': 'M*_by_column',
                                        'z_colnames': ['z_med'],
                                        'z_columns': [3],
                                        'M*_colnames': ['M*_lo', 'M*_hi', 'M*_med'],
                                        'M*_columns': [0, 1, 2],
                                        'data_colnames': ['Re_med_{}', 'dRe_med_{}+'],
                                        'marker': 'D',
                                        'wavelength': 0.5,
                                        'short_title': 'HSC',
                                        'suptitle': 'Kawinwanichakij et. al. 2021, HSC',
                                    },
                                    },

                   'Re_vs_z': {
    'missing_values': '--',
    'filling_values': 0.0,
    'reader': read_Re_vs_z_Mstar_fits,
    'plotter': plot_Re_vs_z_Mstar_fits,
    'x-values': 'M*med',
    'y-values': ['B_{}', 'beta_{}'],
    'y-errors+': ['dB_{}+', 'dbeta_{}+'],
    'y-errors-': ['dB_{}-', 'dbeta_{}-'],
    'xlabels': ['$\\bm{\\log_{10}(M^*/M_\\odot)}$',
                '$\\bm{\\log_{10}(M^*/M_\\odot)}$'],
    'ylabels': ['$\\bm{B}$ (kpc)', r'$\bm\beta$'],
    'titles': ['$\\bm{B}$: $\\bm{R_e = B*(1 + z)^{-\\beta}}$',
               '$\\bm{\\beta}$: $\\bm{R_e = B*(1 + z)^{-\\beta}}$'],
    'xlabel': '$\\bm{z}$',
    'ylabel': '$\\bm{\\log10(R_{e}}/\\mathrm{kpc})$',
    'martorano_2024': {
        'samples': ['All', 'Starforming', 'Quiescent'],
        'colors': ['black', 'darkslateblue', 'firebrick'],
        'filename': 'martorano_2024_table2_Re_vs_z.txt',
        'skip_header': 4,
        'x_colnames': ['M*_lo', 'M*_hi', 'M*med'],
        'x_columns': [0, 1, 2],
        'data_colnames': ['alpha_{}', 'dalpha_{}+', 'dalpha_{}-',
                          'beta_{}', 'dbeta_{}+', 'dbeta_{}-',
                          ],
        'flip_sign': True,
        'flip_value_columns': 3,
        'convert_from_log': True,
        'value_columns': [0],
        'error_columns': [1],
        'error_n_columns': [2],
        'marker': 's',
        'wavelength': 1.5,
        'short_title': 'JWST+COSMOS',
        'suptitle': 'Martorano et. al. 2024, JWST+COSMOS',
    },
    'george_2024_3000': {
        'samples': ['Starforming', 'Quiescent'],
        'colors': ['mediumslateblue', 'crimson'],
        'filename': 'George_2024_Table3a_3000_Re_vs_z.txt',
        'skip_header': 5,
        'x_colnames': ['M*med'],
        'x_columns': [0],
        'data_colnames': ['B_{}', 'dB_{}+',
                          'beta_{}', 'dbeta_{}+',
                          ],
        'flip_sign': True,
        'flip_value_columns': 2,
        'marker': 'X',
        'wavelength': 0.3,
        'short_title': 'CLAUDS+HSC',
        'suptitle': 'George et. al. 2024, CLAUDS+HSC',
    },
    'george_2024_5000': {
        'samples': ['Starforming', 'Quiescent'],
        'colors': ['mediumslateblue', 'crimson'],
        'filename': 'George_2024_Table3b_5000_Re_vs_z.txt',
        'skip_header': 5,
        'x_colnames': ['M*med'],
        'x_columns': [0],
        'data_colnames': ['B_{}', 'dB_{}+',
                          'beta_{}', 'dbeta_{}+',
                          ],
        'flip_sign': True,
        'flip_value_columns': 2,
        'marker': 'P',
        'wavelength': 0.5,
        'short_title': 'CLAUDS+HSC',
        'suptitle': 'George et. al. 2024, CLAUDS+HSC',
    },
    'kawinwanichakij_2021': {
        'samples': ['All', 'Quiescent', 'Starforming'],
        'colors': ['black', 'firebrick', 'royalblue'],
        'filename': 'kawinwanichakij_Table6b_Re_vs_z.txt',
        'skip_header': 2,
        'x_colnames': ['M*_lo', 'M*_hi', 'M*med'],
        'x_columns': [0, 1, 2],
        'data_colnames': ['B_{}', 'dB_{}+',
                          'beta_{}', 'dbeta_{}+',
                          ],
        'marker': 'D',
        'wavelength': 0.5,
        'short_title': 'HSC',
        'suptitle': 'Kawinwanichakij et. al. 2021 -- HSC',
    },
    'mowla_2019': {
        'samples': ['All', 'Starforming', 'Quiescent'],
        'colors': ['black', 'mediumblue', 'crimson'],
        'filename': 'mowla_2019_Table1b_3b_Re_vs_z.txt',
        'skip_header': 3,
        'x_colnames': ['M*med'],
        'x_columns': [0],
        'data_colnames': ['B_{}', 'dB_{}+', 'beta_{}', 'dbeta_{}+'],
        'marker': 'o',
        'wavelength': 0.55,
        'short_title': 'COSMOS-DASH',
        'suptitle': 'Mowla et al. 2019, COSMOS-DASH',
    },
    'vanderWel_2014': {
        'samples': ['Starforming', 'Quiescent'],
        'colors': ['dodgerblue', 'orangered'],
        'filename': 'vanderWel_2014_table2_Re_vs_z.txt',
        'skip_header': 5,
        'x_colnames': ['M*med'],
        'x_columns': [0],
        'data_colnames': ['alpha_{}', 'dalpha_{}+', 'beta_{}', 'dbeta_{}+'],
        'marker': '^',
        'flip_sign': True,
        'flip_value_columns': 2,
        'convert_from_log': True,
        'value_columns': [0],
        'error_columns': [1],
        'wavelength': 0.5,
        'short_title': '3D-HST+CANDELS',
        'suptitle': 'van der Wel et al. 2014, 3D-HST+CANDELS',
    },
},
}


def read_size_data(data, authors, validation_info=validation_info, info_key='Re_vs_z', sample='',
                   size_dir=SIZE_DATA_DIR):
    # setup data dict
    if info_key not in data.keys():
        data[info_key] = {}
    v_info = validation_info[info_key]
    for author in authors:
        if author in v_info.keys():
            if author not in data[info_key].keys():
                data[info_key][author] = {}
            fn = os.path.join(
                size_dir,
                author,
                v_info[author]['filename'].format(sample))
            print('Reading data from {}'.format(fn))
            tmp = np.genfromtxt(fn, skip_header=v_info[author]['skip_header'],
                                missing_values=v_info['missing_values'],
                                filling_values=v_info['filling_values'],
                                )
            if info_key == 'Re_vs_Mstar_data':  # separate files for each galaxy type
                data[info_key][author] = v_info['reader'](
                    tmp, data[info_key][author], author, v_info, sample=sample)
            elif 'Re_vs_z' in info_key or 'Re_vs_Mstar' in info_key:
                data[info_key][author] = v_info['reader'](
                    tmp, data[info_key][author], author, v_info)

    return data
