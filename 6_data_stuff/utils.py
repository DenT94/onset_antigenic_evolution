import random
import numpy.random as rnd
# import networkx as nx
# from anytree import Node
from tqdm import tqdm
import numpy as np
import os
import sys
import statsmodels.api as sm
if(sys.version_info[1]<= 7):
    import pickle5 as pickle
else:
    import pickle

import pandas as pd
import scipy.interpolate as spi
import scipy.optimize as so

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from matplotlib.ticker import FuncFormatter

fmt = lambda x, pos: '{:.1f}'.format(x)
from numba import jit

#matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
plt.rc('mathtext', default='regular')
plt.rc('lines', linewidth=3.0)
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['axes.labelsize']=35
plt.rcParams['legend.fontsize']= 35
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['image.cmap']='coolwarm'
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['figure.titlesize'] = 40
#set legend titlesize to 40
plt.rcParams['legend.title_fontsize'] = 40

sys.path.append('..')
from src.evo_eq_model import *

data_folder = os.getcwd()+'/data/'
os.makedirs(data_folder, exist_ok = True)

saves_folder = os.getcwd()+'/saves/'
os.makedirs(saves_folder, exist_ok = True)

home = os.path.expanduser("~")
project_path =  os.path.relpath("..")
if project_path not in sys.path:
    sys.path.append(project_path)

sys.path.insert(1, project_path)
output_folder= project_path+'/outputs/'

def model(X, t, R0, kappa):
    x = X[0]
    y = X[1]
    z = X[2]
    dxdt = - R0/(1+kappa * y) *x * y
    dydt =  R0/(1+kappa* y) * x * y - y
    dzdt = y
    return [dxdt, dydt, dzdt]

def model_w_params(R0, N, k): 
    def model(y,t):
        S = y[0]
        I = y[1]
        R0_eff = R0/(1+k*I/(N))
        dSdt = - R0_eff * S*I/N 
        dIdt =  R0_eff * S*I/N -I
        return [dSdt,dIdt]
    return model

@jit(nopython=True)    
def find_t_intersection(f,g, t):
    #print(np.sign(f - g))
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    #print(idx)
    if len(idx)>0:
        return t[idx[0]]
    else:
        return np.inf

@jit(nopython=True)    
def find_ind_intersection(f,g):
    #print(np.sign(f - g))
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    #print(idx)
    if len(idx)>0:
        return idx[0]
    else:
        return np.inf

@jit(nopython=True)
def get_p_surv_inv_int(fit, dt,reg= 1e-1):
    L = len(fit)
    pi_ext_ode_hand_inv_t = np.zeros(L)
    pi_ext_ode_hand_inv_t[-1] = np.minimum(1/(1+fit[-1]),1-reg)
    for i in range(L-1):
        pi_ext= pi_ext_ode_hand_inv_t[L-1-i]
        pi_ext_ode_hand_inv_t[L-2-i] = pi_ext -dt * (1-pi_ext)*(-1+pi_ext*(1+fit[L-2-i]))
    p_surv = 1-pi_ext_ode_hand_inv_t
    return p_surv

def find_x_inf(R0,k):
    if k>0:
        c = (R0-k+R0*k)/(k*(R0-k))
        rho = 1/(R0/k)
        y_x = lambda x: c*x**rho + x/(rho-1) - 1/k

        roots = so.fsolve(y_x, 0)
        return roots[0]
    else:
        return np.real(-1/R0*sps.lambertw(-R0*np.exp(-R0)))
    
def integrate_trajectory(R0,kappa,N, I0=10,Nsteps = 10000):
    S0 = N-I0
    y0 = [S0,I0]
    x_inf = find_x_inf(R0,kappa)
    
    t_end = 1.5*np.log(N)* (1/(R0-1) + 1/(1-R0*x_inf))
    ts = np.linspace(0, t_end,Nsteps)
    dt = ts[1]-ts[0]
    solution  = odeint(model_w_params(R0,N,kappa),y0,ts).T
    x,y= solution
    tp = ts[np.argmax(y)]


    return ts, solution


def get_clade_stats_dataframe(folder_name, reference_date, x_thresh_vals = [0,1e-5,1e-4,1e-3,1e-2,1e-1]):
    if os.path.exists(saves_folder+folder_name):
        clade_stats = pd.read_feather(saves_folder+folder_name)
    else:
        clade_folder = data_folder + '/' +  folder_name + '/'
        assert os.path.exists(clade_folder)

        clade_statistics = pd.read_csv(clade_folder + 'clade_statistics.tsv', sep = '\t')
        clade_stats = pd.DataFrame(columns = ['Clade','Max_Freq','Orig_Time'])
        
        clade_stats['Clade'] = clade_statistics['Clade']
        clade_stats['Max_Freq'] = clade_statistics.groupby('Clade')['Sublineage_Freq'].transform('max')
        clade_stats['Orig_Time'] = clade_statistics.groupby('Clade')['Time'].transform('min')
        clade_stats = clade_stats.drop_duplicates()
        for x_th in tqdm(x_thresh_vals):
            clade_stats[f'Time_x_bgr_{x_th}'] = clade_statistics[clade_statistics['Sublineage_Freq']>x_th].groupby('Clade')['Time'].transform('min')
            clade_stats[f'day_diff_x_bgr_{x_th}'] = (pd.to_datetime(clade_stats[f'Time_x_bgr_{x_th}']) - reference_date).dt.days
        clade_stats['day_diff'] = (pd.to_datetime(clade_stats['Orig_Time']) - reference_date).dt.days
        clade_stats.to_feather(saves_folder+folder_name)

    return clade_stats

def get_kde(data, bw_adjust = .15, bw_method = 'scott', ax = None, grid_min = None, grid_max = None, grid_size = 1000):
    #if ax is None create a fake figure
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit(bw=bw_adjust)
    # Define the grid range if not specified
    if grid_min is None:
        grid_min = np.min(data)
    if grid_max is None:
        grid_max = np.max(data)
    
    # Create a fixed grid
    grid = np.linspace(grid_min, grid_max, grid_size)
    
    # Evaluate the KDE on the fixed grid
    kde_values = kde.evaluate(grid)
    
    return grid, kde_values

def get_covid_data_World(reference_date):
    covid_data = pd.read_csv(data_folder + '/covid_data.csv')
    covid_data_World  = covid_data[covid_data['location'] == 'World']#.query(f'date < "{last_date}"')

    covid_data_World['Time_datetime'] = pd.to_datetime(covid_data_World['date'])
    covid_data_World['day_diff'] = covid_data_World['Time_datetime']-reference_date

    covid_data_World['day_diff'] = covid_data_World['day_diff'].apply(lambda x: x.days)
    covid_data_World['weekly_new_cases_smoothed'] = covid_data_World['new_cases_smoothed']
    return covid_data_World

def get_df_reworked(fname = 'fitness_france_newest.txt',reference_date = pd.to_datetime('2021-01-01')):

    fitness_france = pd.read_csv(data_folder+ '/'+ fname,sep = '\t')

    fitness_france['day_diff'] = (pd.to_datetime(fitness_france['date']) - reference_date).dt.days

    clade_list = ['WT', 'B.1.617.2', 'B.1.1.7', 'BA.1', 'BA.2', 'BA.5', 'JN.1', 'JN.1.4', 'BF.7', 'XBB.1.5', 'XBB.1.16', 'HK.3'  ]
    avg_fit = np.sum([fitness_france[c+'_freq']*(fitness_france[c+'_fitness_inf']+fitness_france[c+'_fitness_vac'])\
                    for c in clade_list],axis=0)

    df_reworked = pd.DataFrame(columns = ['clade','day_diff','freq','avg_fit'])

    for c in clade_list:
        vals  =fitness_france[[c+'_freq', c+'_fitness_vac', c+'_fitness_inf', c + '_F_pot_inf', c + '_F_pot_vac' , 'day_diff']]
        df_vals = pd.DataFrame(vals.values,columns=['freq','fit_vac','fit_inf',  'f_pot_inf','f_pot_vac','day_diff'])
        df_vals['avg_fit'] = avg_fit
        df_vals['cases'] = fitness_france['cases']
        extra_col = [c for i in range(len(vals))]

        df_reworked = pd.concat([df_reworked,pd.concat( [pd.DataFrame(extra_col,columns=['clade']),df_vals],axis=1)],axis=0)
    df_reworked['fit'] = df_reworked['fit_vac'] + df_reworked['fit_inf']
    df_reworked['f_pot'] = df_reworked['f_pot_inf'] + df_reworked['f_pot_vac']

    df_reworked['selection'] = (df_reworked['fit'] - df_reworked['avg_fit'])*(df_reworked['freq']>0)
    df_reworked['pot_selection'] = (df_reworked['f_pot'] - df_reworked['avg_fit'])*(df_reworked['freq']>0)
    df_reworked['y_t'] = df_reworked['freq']*df_reworked['cases']
    df_reworked['s_times_y_t'] = df_reworked['y_t']*df_reworked['selection']
    df_reworked['pot_s_times_y_t'] = df_reworked['y_t']*df_reworked['pot_selection']

    return df_reworked








