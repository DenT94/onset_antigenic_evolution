#import src path 
import sys
# sys.path.append('src')
sys.path.append('../src')

# Import src modules

import evo_eq_model_new as neem

import argparse
import logging
import scipy.stats as stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from matplotlib.lines import Line2D

from scipy.signal import savgol_filter

from tqdm import tqdm

from scipy.linalg import svd

import pickle as pickle
import os

import matplotlib.colors as mcolors

from itertools import product
from dask import delayed, compute
from dask.diagnostics import ProgressBar

import time 

import pandas as pd
from itertools import product

import numpy.random as rnd

###################################################################################

def get_ready_df(number_of_seeds = 70):
    N_vals = [1e8]
    R0_vals = [2.5]
    yc_vals = np.logspace(-4.5,0,30)
    nu_vals = [1]
    mu_vals = np.logspace(0, np.log10(700), 10)/np.average(N_vals)
    p_alpha_vals = np.array([0.1])
    p_alpha = p_alpha_vals[0]
    mu_bd_vals = [0]
    seeds =rnd.randint(0, 1000000, number_of_seeds, dtype=int)
    dimensionalities = [np.inf]


    # xi_s= [3/p_alpha, 3/(2*p_alpha)]
    xi_s = [20/p_alpha, 10/p_alpha]

    params = list(product(R0_vals, yc_vals, nu_vals, mu_vals, p_alpha_vals, xi_s, N_vals, seeds))

    pre_df = pd.DataFrame(params, columns = ['R0', 'yc', 'nu', 'mu', 'p_alpha', 'xi', 'N', 'seed'])

    pre_df['dbar_over_xi'] = 1/(pre_df['xi']*pre_df['p_alpha'])
    pre_df['muN_yc_times_dbar_over_xi'] = pre_df['mu']*pre_df['N']* pre_df['yc']* pre_df['dbar_over_xi']
    post_df = pre_df#.query('1e-1 < muN_over_kappa_times_dbar_over_xi < 10')
    post_df = post_df.reset_index(drop=True)
    return post_df

def get_observables(R0, yc, nu, mu, p_alpha, xi, N,
                    seed, dt = .5, snapshot_interval=10,
                    progress_bar = False,
        ):

    E = neem.eqModel(N, R0, yc, nu, mu, p_alpha, xi, initial_infected=10, figures_folder= None, seed = seed, dt = dt)

    start_time = time.time()

    outcome = E.run_to_extinction(snapshot_interval=snapshot_interval, progress_bar=progress_bar)

    elapsed = time.time() - start_time

    conditioned_strains = [x for x in E.strain_set if len(x.times)>0]
    if E.t_intersection == np.inf:
        # No strains survived
        return "extinct", np.inf, np.inf, np.nan, 0, elapsed
    
    intersecting_strains = [x for x in conditioned_strains if x.root_intersection_time < np.inf]
    # if len(intersecting_strains) == 0:
    #     # No strains intersected
    #     return "extinct", np.inf, np.nan, 0
    # else:
    intersecting_strain = intersecting_strains[0]
        
    t_birth = intersecting_strain.birth_time
    t_intersection = intersecting_strain.root_intersection_time
    d_intersection = intersecting_strain.distance_to_root
    n_contenders_for_intersection = E.n_contenders_for_intersection
    outcome = "extinct" if E.t_intersection == np.inf else "success"
    
    return outcome, t_birth, t_intersection, d_intersection, n_contenders_for_intersection, elapsed


def dummy_get_observables(R0, yc, nu, mu, p_alpha, xi, mu_bd, N, seed, dt = .1,max_mutants=500):
    return 1, 1, 1, 1, 1, 1

def setup_logging(log_file='dask_run.log'):
    # configure root logger
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    # set dask to log with the root logger
    logging.getLogger('dask').setLevel(logging.DEBUG)

    # also log to console if needed
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_cores', type=int, default=11, help='Number of cores to use')
    parser.add_argument('--num_seeds', type=int, default=70, help='Number of seeds to use')
    parser.add_argument('--dummy', type=bool, default=False, help='Use dummy function')
    parser.add_argument('--max_mutants', type=int, default=100, help='Number of mutants to use')
    parser.add_argument('--filename', type=str, default='', help='Filename to add')

    args = parser.parse_args()
    logging.info(f"starting time: {pd.Timestamp.today()}")
    start_time = time.time()

    if args.dummy:
        logging.info('Using dummy function')

    from dask import delayed, compute
    from dask.distributed import Client, LocalCluster
    from dask.diagnostics import ProgressBar

    # Set the number of cores/workers you want to use
    num_cores = 11  # Replace with the number of cores you wish to use

    # Set up a local Dask cluster with the specified number of workers
    cluster = LocalCluster(n_workers=num_cores, threads_per_worker=1, dashboard_address=':8787')
    client = Client(cluster)

    # Print the dashboard URL
    logging.info(f"Dask dashboard available at: {client.dashboard_link}")


    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    tomorrow = pd.Timestamp.today().replace(day=pd.Timestamp.today().day+1).strftime('%Y-%m-%d')

    df_observables = get_ready_df(number_of_seeds = args.num_seeds)
    observables = ['outcome', 't_birth', 't_intersection', 'd_intersection', 'n_contenders_for_intersection', 'elapsed']

    for obs in observables:
        df_observables[obs] = np.nan
    

    # Create delayed tasks for parallel computation
    tasks = []
    if args.dummy:
        for index, row in df_observables.iterrows():
            task = delayed(dummy_get_observables)(
                row['R0'], row['yc'], row['nu'], row['mu'], row['p_alpha'],
                row['xi'], row['mu_bd'], row['N'], row['seed'], max_mutants=100
            )
            tasks.append(task)
    else:
        #def get_observables(R0, yc, nu, mu, p_alpha, xi, N,
        #             seed, dt = .1, snapshot_interval=20,
        #             progress_bar = False,
        # )
        for index, row in df_observables.iterrows():
            task = delayed(get_observables)(
                row['R0'], row['yc'], row['nu'], row['mu'], row['p_alpha'],
                row['xi'], row['N'], int(row['seed']))
            tasks.append(task)

    
    results = client.compute(tasks, sync=True)
    # Unpack results and assign to df_observables
    for i, res in enumerate(results):
        df_observables.loc[i, [
            'outcome', 't_birth', 't_intersection', 'd_intersection', 'n_contenders_for_intersection', 'elapsed'
        ]] = res

    df_observables['dbar_over_xi'] = 1 / (df_observables['xi'] * df_observables['p_alpha'])
    
    if args.dummy:
        i=0
        while os.path.exists(today + 'new_df_observables_'+args.filename+'_dummy.feather'):
            today = pd.Timestamp.today().strftime('%Y-%m-%d')+ '_' + str(i) + '_'
            i+=1
        df_observables.to_feather(today + 'new_df_observables_'+args.filename+'_dummy.feather')
        logging.info('Saved ' + today + 'new_df_observables_'+args.filename+'_dummy.feather')
    else:
        i=0
        while os.path.exists(today + 'new_df_observables_'+args.filename+'.feather'):
            today = pd.Timestamp.today().strftime('%Y-%m-%d')+ '_' + str(i) + '_'
            i+=1
        df_observables.to_feather(today + 'new_df_observables_'+args.filename+'.feather')
        logging.info('Saved ' + today + 'new_df_observables_'+args.filename+'.feather')

    logging.info(f"Client status: {client.status}")
    logging.info(f"Scheduler address: {client.scheduler.address}")
    # Close the client when done
    client.close()
    logging.info(f"elapsed time: {time.time() - start_time}")
    

if __name__ == '__main__':
    main()