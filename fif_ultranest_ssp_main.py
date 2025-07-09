#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:25:23 2022

@author: fabioditrani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:02:48 2022

@author: fabioditrani
"""
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import fif_ultranest_ssp_class as myf
from fif_ultranest_ssp_class import fif_ultranest


import corner
from glob import glob
import sys
#plt.style.use('classic')
plt.style.use('default')
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ultranest
from ultranest.plot import PredictionBand
import h5py
from astropy.cosmology import Planck18 as cosmo  # Modello cosmologico di Planck 2018


import gc

def eta_universo(z):
    """ Restituisce l'età dell'Universo in miliardi di anni a un dato redshift z """
    return cosmo.age(z).value 
home = '../'
alpha_str = ['m02','p00','p02','p04','p06']
alpha = np.array([-0.2,0,0.2,0.4,0.6])
zhlist = ['m1.79','m1.49','m1.26','m0.96','m0.66','m0.35','m0.25','p0.06','p0.15','p0.26']
zh = np.array([-1.79,-1.49,-1.26,-0.96,-0.66,-0.35,-0.25,0.06,0.15,0.26])
isoalpha = ['0.00','0.00','0.00','0.40','0.40']

agelist_ssp = []
aa = np.sort(glob('../../models/sMILES_SSPs/Universal_Kroupa/aFep00/*1.30*'))
age = np.zeros([int(len(aa)/len(zh))],dtype = float)
age_ssp = np.zeros([int(len(aa)/len(zh))],dtype = float)
#Ech1.30Zm0.25T00.0300_iTp0.00_baseFe
for i in range(len(age)):
    #agelist.append(name_models[i][37:-15])
    agelist_ssp.append(aa[i][63:-28])
    age[i] = float(agelist_ssp[i])
    age_ssp[i] = float(agelist_ssp[i])

hdu = fits.open('../../models/sMILES_SSPs/Universal_Kroupa/aFep00/Mku1.30Z'+zhlist[0]+'T'+agelist_ssp[0]+'_iTp'+isoalpha[0]+'_ACFep00_aFep00.fits')
h2 = hdu[0].header
lambda_models = h2['CRVAL1'] + h2['CDELT1'] * np.arange(h2['NAXIS1'])
flux_models = np.zeros([len(age),len(zh),len(alpha_str),len(lambda_models)])
for i in range(len(age)):
    for j in range(len(zh)):
        for k in range(len(alpha_str)):
            hdu = fits.open('../../models/sMILES_SSPs/Universal_Kroupa/aFe'+alpha_str[k]+'/Mku1.30Z'+zhlist[j]+'T'+agelist_ssp[i]+'_iTp'+isoalpha[k]+'_ACFep00_aFe'+alpha_str[k]+'.fits')
            flux_models[i,j,k,:] = hdu[0].data



sys.exit()




home = '../../spectra/desi/'
name = 'stacked_50a50_mass_red01_median_deltasel_DR1_113115'
with h5py.File(home+'/'+name+'.h5', 'r') as hf:
    # Accesso agli array principali
    mass_data = hf[name+'/mass'][:]
    redshift_data = hf[name+'/redshift'][:]
    flag_stack_data = hf[name+'/flag_stack'][:]
    wave_data = [hf[name+f'/wavelength/array_{i}'][:] for i in range(len(hf[name+'/wavelength']))]
    flux_data = [hf[name+f'/flux/array_{i}'][:] for i in range(len(hf[name+'/flux']))]
    error_data = [hf[name+f'/error/array_{i}'][:] for i in range(len(hf[name+'/error']))]



c = 299792.458
np.random.seed(42)
ndim = 3

fwhm_mod = 2.51
res_instr = 1400

velsigma = np.loadtxt('../../spectra/desi/velsigma_'+name+'.txt',unpack = True)

sys.exit()
blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx = np.loadtxt('def_indices.dat',usecols = (1,2,3,4,5,6), unpack = True)
name_ind = np.loadtxt('def_indices.dat',usecols = (0),unpack = True,dtype = str)
name_ind = np.array(name_ind)

ind_selected = np.where( (name_ind == 'Dn4000')   |  (name_ind == 'HgammaF')| (name_ind == 'CaH') | (name_ind == 'CaK') | (name_ind == 'HdeltaF')| (name_ind == 'Fe4383') #
                          |  (name_ind == 'Hbo') |  (name_ind == 'Mgb') |  (name_ind == 'Fe5270') |  (name_ind == 'Fe5335') 
                        )[0]
#
blue_sx = blue_sx[ind_selected]
blue_dx = blue_dx[ind_selected]
feat_sx = feat_sx[ind_selected]
feat_dx = feat_dx[ind_selected]
red_sx = red_sx[ind_selected]
red_dx = red_dx[ind_selected]
name_ind = name_ind[ind_selected]


sys.exit()
i = 0
lambda_rest = wave_data[i]/(1+velsigma[0][i]/c)
flux_gal = flux_data[i].squeeze()
error_gal = error_data[i].squeeze()

nsimul = 101
distr_err = np.random.standard_normal(size = (len(lambda_rest),nsimul))

 
flux_2D = np.tile(flux_gal, (nsimul,1)).transpose()
error_flux_2D = np.tile(error_gal, (nsimul,1)).transpose()
error_flux_2D[:,0] = 0.
flux_sim = distr_err*error_flux_2D+flux_2D
 
output = myf.calcindex_all(lambda_rest, flux_sim.transpose(), blue_sx, blue_dx, feat_sx, feat_dx, red_sx, red_dx, name_ind)
ind_obs = output[:,0]
error_indices = np.std(output,axis = 1)
 
 
sel_ind = np.where( (name_ind == 'Dn4000'))[0]
sel_fif = np.delete(np.linspace(0,len(ind_selected)-1,len(ind_selected)),sel_ind).astype(int)
 

param_names = ['Age','[Z/H]','alpha']

fwhm_mod = 2.51

age_uni = eta_universo(redshift_data[i])
call_fif_ultranest = fif_ultranest(lambda_rest,flux_gal,error_gal,flux_sim,ind_obs,error_indices,velsigma[1][i],lambda_models,flux_models,age_uni, age, zh,alpha,
         blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx,name_ind,res_instr,fwhm_mod,sel_fif,sel_ind,nsimul)
 
t0 = time.time()
log_dir = 'results'
sampler = ultranest.ReactiveNestedSampler(param_names, call_fif_ultranest.log_likelihood_ultranest,call_fif_ultranest.my_prior_transform,log_dir=log_dir+'/run'+str(i),vectorized = True)#
  
result = sampler.run(dlogz = 0.3,min_num_live_points=500,show_status = None,viz_callback = False)#
sampler.print_results()
sampler.plot_trace()
sampler.plot()
sampler.plot_run()
t1 = time.time()
print(t1-t0)
plt.close('all')
 
 
result_data = np.array(result['samples'])
param_med = np.percentile(result_data,axis = 0,q = [16,50,84])
   
  
figure = corner.corner(result_data,quantiles=[0.16,0.50, 0.84],levels = (0.68,0.95),
                     labels=param_names, show_titles=True, quiet=True,title_fmt = '.2f')
  
print(f'Evidence: {result["logz"]:.2f}')


plt.savefig(log_dir+'/run'+str(i)+'/run1/plots/corner_all.pdf',bbox_inches = 'tight') 
plt.close(figure)
plt.clf()
  

nonan = np.where(np.isfinite(call_fif_ultranest.flux_obs[call_fif_ultranest.sel_obs]))[0]
band = PredictionBand(call_fif_ultranest.lambda_obs_cut[nonan])
band2 = PredictionBand(call_fif_ultranest.lambda_obs_cut[nonan])
besto = np.zeros([len(result['samples'][:,0]),flux_models.shape[-1]])
besto2 = np.zeros([len(result['samples'][:,0]),flux_models.shape[-1]])
indo = 0
for age_int, zh_int,alpha_int in result['samples']:
      besto[indo,:] = myf.interpmodel_3par(zh_int,age_int,alpha_int,zh,age,alpha,flux_models)
      besto2[indo,:] = myf.interpmodel(zh_int,age_int,zh,age,flux_models[:,:,1,:])
      indo += 1
besto_conv = myf.varsmooth_vec(call_fif_ultranest.lambda_models_cut,besto[:,call_fif_ultranest.sel_mod],call_fif_ultranest.sigma_fin_cut)
conv_rebin_besto = myf.vectorized_interp(call_fif_ultranest.lambda_obs_cut,call_fif_ultranest.lambda_models_cut,besto_conv)
 
besto_conv_2 = myf.varsmooth_vec(call_fif_ultranest.lambda_models_cut,besto2[:,call_fif_ultranest.sel_mod],call_fif_ultranest.sigma_fin_cut)
conv_rebin_besto_2 = myf.vectorized_interp(call_fif_ultranest.lambda_obs_cut,call_fif_ultranest.lambda_models_cut,besto_conv_2)

for indo in range(result['samples'].shape[0]):
       band.add(conv_rebin_besto[indo,:][nonan]*np.nansum(call_fif_ultranest.flux_obs[call_fif_ultranest.sel_obs][nonan][100:-100])/np.nansum(conv_rebin_besto[indo,:][nonan][100:-100]))
 
for indo in range(result['samples'].shape[0]):
       band2.add(conv_rebin_besto_2[indo,:][nonan]*np.nansum(call_fif_ultranest.flux_obs[call_fif_ultranest.sel_obs][nonan][100:-100])/np.nansum(conv_rebin_besto_2[indo,:][nonan][100:-100]))

 
flux_obs_cut = flux_gal[call_fif_ultranest.sel_obs]
err_obs_cut = error_gal[call_fif_ultranest.sel_obs]
best_model_conv_rebin = band.get_line(q = 0.5)
best_model_conv_rebin2 = band2.get_line(q = 0.5)

plt.figure(figsize=(30, 15))
for n, ticker in enumerate(name_ind[sel_fif]):#[np.isfinite(ind_obs[sel_fif]) == True]
      n = int(np.where(name_ind[sel_fif] == ticker)[0])
      ax = plt.subplot( 2,int(len(name_ind[sel_fif])/2+1), n + 1)
      ax.set_box_aspect(1)
      ax.set_ylabel(r'Flux/$\AA$')
      selcont = np.concatenate((call_fif_ultranest.selblue[n],call_fif_ultranest.selred[n]))
      selblue = call_fif_ultranest.selblue[n]
      selred = call_fif_ultranest.selred[n]

          
      iniblue = call_fif_ultranest.iniblue[n]
      ifiblue = call_fif_ultranest.ifiblue[n]
      inired = call_fif_ultranest.inired[n]
      ifired = call_fif_ultranest.ifired[n]

      
      meanblue_obs = np.nansum(flux_obs_cut[iniblue:ifiblue+1]*call_fif_ultranest.dblue_vec[n])/call_fif_ultranest.titblue_vec[n]
      meanred_obs = np.nansum(flux_obs_cut[inired:ifired+1]*call_fif_ultranest.dred_vec[n])/call_fif_ultranest.titred_vec[n]

      meanbluex = call_fif_ultranest.meanbluex[n]
      meanredx = call_fif_ultranest.meanredx[n]
      
      meanblue = np.sum(best_model_conv_rebin[iniblue:ifiblue+1]*call_fif_ultranest.dblue_vec[n])/call_fif_ultranest.titblue_vec[n]
      meanred = np.sum(best_model_conv_rebin[inired:ifired+1]*call_fif_ultranest.dred_vec[n])/call_fif_ultranest.titred_vec[n]
  
      m = (meanblue-meanred)/(meanbluex-meanredx)
      q = (meanbluex*meanred-meanredx*meanblue)/(meanbluex-meanredx)

      
      mobs = (meanblue_obs-meanred_obs)/(meanbluex-meanredx)
      qobs = (meanbluex*meanred_obs-meanredx*meanblue_obs)/(meanbluex-meanredx)

      ycontall = m*call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1]+q
      
      meanblue2 = np.sum(best_model_conv_rebin2[iniblue:ifiblue+1]*call_fif_ultranest.dblue_vec[n])/call_fif_ultranest.titblue_vec[n]
      meanred2 = np.sum(best_model_conv_rebin2[inired:ifired+1]*call_fif_ultranest.dred_vec[n])/call_fif_ultranest.titred_vec[n]
  
      m2 = (meanblue2-meanred2)/(meanbluex-meanredx)
      q2 = (meanbluex*meanred2-meanredx*meanblue2)/(meanbluex-meanredx)


      ycontall2 = m2*call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1]+q2

      ycontall_obs = mobs*call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1]+qobs

      ywithcont = flux_obs_cut[selblue[0]:selred[-1]+1]/ycontall_obs
      errorcontall_obs = abs(err_obs_cut[selblue[0]:selred[-1]+1]/ycontall_obs)
      errorcontall_obs[call_fif_ultranest.selfeat[n] -selblue[0]] = call_fif_ultranest.errfeat_obs[n]
      divider = make_axes_locatable(ax)
      ax2 = divider.append_axes("bottom", size="30%", pad=0,sharex = ax)
      ax.figure.add_axes(ax2)
      
      ax.axvline(feat_sx[sel_fif][n],linestyle = 'dashed',color = 'brown')
      ax.axvline(feat_dx[sel_fif][n],linestyle = 'dashed',color = 'brown')
      ax.fill_between(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1], ywithcont-errorcontall_obs, ywithcont+errorcontall_obs,alpha = 0.5)
      ax.plot(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1],ywithcont, color = 'k')
      
      #ax.plot(lambda_obs_cut[selfeat],yfeat, color = 'r',linewidth = 2)
      if n in call_fif_ultranest.ind_selected1:
          ax.plot(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1],best_model_conv_rebin[selblue[0]:selred[-1]+1]/ycontall, color = 'r',linewidth = 2)
          ax2.plot(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1], ywithcont-best_model_conv_rebin[selblue[0]:selred[-1]+1]/ycontall, color="crimson")
          #ax2.plot(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1], ((ywithcont/(best_model_conv_rebin[selblue[0]:selred[-1]+1]/ycontall))-1)*100, color="green")
         #ax.plot(lambda_obs_cut[selblue[0]:selred[-1]+1],conv_rebin_besto[selblue[0]:selred[-1]+1]/ycontall, color = 'g',linewidth = 2)
      if n in call_fif_ultranest.ind_selected2:
          ax.plot(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1],best_model_conv_rebin2[selblue[0]:selred[-1]+1]/ycontall2, color = 'r',linewidth = 2)
          ax2.plot(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1], ywithcont-best_model_conv_rebin2[selblue[0]:selred[-1]+1]/ycontall2, color="crimson")
      
      ax2.fill_between(call_fif_ultranest.lambda_obs_cut[selblue[0]:selred[-1]+1], -errorcontall_obs,errorcontall_obs,alpha = 0.5)
      
      ax.axvline(4861.33)
      #ax.set_xlim(feat_sx[sel_fif][n],feat_dx[sel_fif][n])
      ax.set_xlim(blue_sx[sel_fif][n],red_dx[sel_fif][n])
      ax.fill_between(call_fif_ultranest.lambda_obs_cut[selblue],100, -100,color = 'k',alpha = 0.4)
      ax.fill_between(call_fif_ultranest.lambda_obs_cut[selred],100, -100,color = 'k',alpha = 0.4)
      ax.set_ylim(0.6*np.nanmin(call_fif_ultranest.fluxfeat_obs[n]),1.4*np.nanmax(call_fif_ultranest.fluxfeat_obs[n]))
      ax2.set_ylim(-5*np.nanmedian(errorcontall_obs),5*np.nanmedian(errorcontall_obs))
      ax2.set_xlabel(r'$\lambda [\AA]$')
      ax.tick_params(axis='y', labelsize=25)
      ax.tick_params(axis='x', labelsize=1)
      ax2.tick_params(axis='x', labelsize=25)
      ax2.tick_params(axis='y', labelsize=15)
      
      #ax.legend(loc = 'best')
      ax.set_title(ticker,fontsize = 25)
      #ax.set_xticks([])
      ax2.xaxis.set_major_formatter(FormatStrFormatter('%g'))
      ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
      ax2.axvline(feat_sx[sel_fif][n],linestyle = 'dashed',color = 'brown')
      ax2.axvline(feat_dx[sel_fif][n],linestyle = 'dashed',color = 'brown')
plt.savefig(log_dir+'/run'+str(i)+'/run1/plots/confr_fif.pdf',bbox_inches = 'tight')
  #plt.savefig('mache.pdf',bbox_inches = 'tight')
plt.clf()
plt.close()
plt.close('all')
del result_data, band, band2, besto, besto2, conv_rebin_besto, conv_rebin_besto_2
del sampler, result, call_fif_ultranest
gc.collect()












