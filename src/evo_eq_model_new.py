import random
import numpy.random as rnd
# import networkx as nx
from anytree import Node
from tqdm import tqdm
import numpy as np
import os
import sys
import time
# from jupyter_server import serverapp as app; 
# import ipykernel, requests;
if(sys.version_info[1]<= 7):
    import pickle5 as pickle
else:
    import pickle

from itertools import product    
import scipy.special as sps
import scipy.optimize as spo
from scipy.integrate import odeint
    
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.stats import gamma

###Random function

def sample_vector(d, scale, shape = 2, rng = None):
    # sample radius from gamma distribution
    if rng is None:
        radius = gamma.rvs(a=shape, scale=scale)
        direction = np.random.normal(size=d)
    else:
        radius = gamma.rvs(a=shape, scale=scale, random_state = rng)
        direction = rng.normal(size=d)

    direction /= np.linalg.norm(direction)  # normalize to lie on the unit sphere

    # scale direction by radius
    vector = radius * direction/shape #divide by shape to ensure that mean is scale
    return vector

def sample_vector_exp(d, scale, rng = None):
    # sample radius from exponential distribution
    if rng is None:
        radius = np.random.exponential(scale)
        direction = np.random.normal(size=d)
    else:
        radius = rng.exponential(scale)
        direction = rng.normal(size=d)

    direction /= np.linalg.norm(direction)  # normalize to lie on the unit sphere

    # scale direction by radius
    vector = radius * direction
    return vector

def sample_vector_geom(d, scale, rng = None):
    if rng is None:
        radius = rng.geometric(1/scale)
        direction = np.random.normal(size=d)
    else:
        radius = rng.geometric(1/scale)
        direction = rng.normal(size=d)

    direction /= np.linalg.norm(direction)  # normalize to lie on the unit sphere

    # scale direction by radius
    vector = radius * direction
    return vector

### control functions

fractional_fun = lambda y,x,yc, R0: 1/(1+(R0-1)*y/yc) 

exp_fun = lambda y,x,yc, R0: np.power(R0, -y/yc)

linear_fun = lambda y,x,yc, R0: 1-(1-1/R0)*y/yc

constant_y_fun = lambda y,x,yc, R0: np.abs(np.minimum(1/(R0*x), 1)) if y/yc>=1 else 1

fractional_fun_modified = lambda y,x,yc, R0: 1/(1+ (R0*x>1)*(R0*x-1)*y/yc) 

control_labels = ['Fractional','Exponential','Linear', 'Constant infected fraction', 'Fractional modified']

control_func_dict = {
    'Fractional': fractional_fun,
    'Exponential': exp_fun,
    'Linear': linear_fun,
    'Constant infected fraction': constant_y_fun,
    'Fractional modified': fractional_fun_modified
}

## classes

class Strain(Node):
    def __init__(self, name, parent=None, children = None, distance_to_parent =0,population=None):
        super().__init__(name, parent=parent, children=children,)
        self.population = population
        self.distance_to_parent = distance_to_parent
        if self.parent is not None:
            self.distance_to_root = self.parent.distance_to_root + self.distance_to_parent
        else:
            self.distance_to_root = self.distance_to_parent
        
        self.recovered_size=0

        if self.name==0:
            pass
            self.active_infected=0
        else:
            self.active_infected=1
        
        if self.parent is None:
            self.susceptible_size = self.population.N - self.active_infected
        else:
            c_pc = np.exp(-self.distance_to_parent/self.population.xi)
            self.susceptible_size = self.parent.susceptible_size +( 1-c_pc)*self.parent.recovered_size - self.active_infected

        self.infected_history = []
        self.susceptible_history = []
        self.fitness_history = []
        self.recovered_history = []
        self.times = []
        self.frequency_history = []
        self.selection_history = []

        self.root_intersection_time = np.inf
        try:
            self.initial_frequency = 1/np.sum([x.active_infected for x in self.population.alive_strains])
        except AttributeError:
            self.initial_frequency = 1
        self.birth_time = self.get_current_time()

    def mutate(self, name, distance_to_parent,population):
        return Strain(parent=self,name=name, distance_to_parent=distance_to_parent, population=population)
    
    def mutate_random(self,name,p,population):
        d_pc = np.abs(sample_vector_geom(1, 1/p, rng = population.rng).squeeze())
        return self.mutate(name, d_pc ,population)
    
    def calc_tree_distance(self, other_strain):
        my_ancestor_set = set(self.ancestors)
        other_ancestor_set = set(other_strain.ancestors)

        cas_my_other = sorted(my_ancestor_set.intersection(other_ancestor_set),key = lambda x: x.depth)
        if not(my_ancestor_set):
            mrca = self
        elif not(other_ancestor_set):
            mrca = other_strain
        else:
            cas_my_other[-1]

        my_ancestors_only = my_ancestor_set.difference(cas_my_other)
        other_ancestors_only =  other_ancestor_set.difference(cas_my_other)

        if my_ancestors_only is None:
            d_self_ca = self.distance_to_parent +np.sum([s.distance_to_parent for s in my_ancestors_only])
        else:
            d_self_ca = self.distance_to_parent
        if other_ancestors_only is None:
            d_other_ca = other_strain.distance_to_parent +np.sum([s.distance_to_parent for s in other_ancestors_only])
        else:
            d_other_ca = other_strain.distance_to_parent
        return d_self_ca + d_other_ca
    
    def tree_distance(self, other_strain):
        if self.name!=other_strain.name:
            try: 
                return self.population.strain_couples[frozenset((self.name,other_strain.name))]
            
            except KeyError:
                
                d_ij = self.calc_tree_distance(other_strain)
                self.population.strain_couples[frozenset((self.name,other_strain.name))] = d_ij
                return d_ij
        else:
            return 0
        
    def calc_infected(self):
        self.infected_history.append(self.active_infected)
    
    def calc_susceptible(self):
        self.susceptible_history.append(self.susceptible_size)
    
    def calc_recovered(self):
        self.recovered_history.append(self.recovered_size)

    def get_current_time(self):
        return self.population.schedule.steps * self.population.dt
    
    def check_intersection(self):
        if self.name!=0 and np.isinf(self.root_intersection_time):
            if self.active_infected> self.root.active_infected:
                self.root_intersection_time = self.get_current_time()

    def calc_susc_for_timestep(self):
        if self.parent is None:
            return self.susceptible_size
        
        root = self.population.root_strain
        S_root = root.susceptible_size
        R_root = root.recovered_size
        descendants_without_self = [d for d in self.population.strain_set if (d.name!=self.name) and (d.name!=self.population.root_strain.name)]
        # descendants_without_self = [d for d in self.root.children if d.name!=self.name]
        c_pc = np.exp(-self.distance_to_root/self.population.xi)

        S = S_root + (1-c_pc) * R_root - np.sum([np.exp(-d.distance_to_root/self.population.xi)*d.recovered_size for d in descendants_without_self]) - self.recovered_size
        return np.maximum(S,0)
    
    def calc_frequency(self):
        Itot= np.sum([x.active_infected for x in self.population.alive_strains])
        self.frequency_history.append(self.active_infected/Itot)
    
    def calc_selection_against_parent(self):
        if self.parent is None:
            self.selection_history.append(0)
        else:
            try:
                self.selection_history.append(self.fitness_history[-1] - self.parent.fitness_history[-1])
            except IndexError:
                self.selection_history.append(0)

    def calc_fitness(self, x_avg):
        Itot = np.sum([x.active_infected for x in self.population.alive_strains])
        R0 = self.population.infection_rate 
        yc = self.population.yc
        N= self.population.N
        
        x= self.calc_susc_for_timestep()/self.population.N
        
        f = R0 *self.population.control_func(Itot/N, x_avg, yc, R0)*x-1
        self.fitness_history.append(f)

class eqModel(Model):
    
    def __init__(self, N, infection_rate, yc, recovery_rate, mutation_rate, p_alpha,xi, initial_infected,
                dt=None, datacoll = False, figures_folder = None, plotflag = False, collect_freqs= False, runmode=None, control_mode = None,mode= None,
                seed = None):
        # self.graph = nx.complete_graph(num_nodes)
        self.schedule = RandomActivation(self)
        self.running=True
        
        self.infection_rate = infection_rate
        self.yc = yc
        self.recovery_rate = recovery_rate
        self.mutation_rate = mutation_rate
        self.p_alpha = p_alpha

        self.xi = xi
        self.dt = dt
        self.N = N
        self.initial_infected = initial_infected

        if runmode is None:
            self.runmode = 'end'
            # print('Mode not initialized. Defaulted to min for the calculation of susceptibles.')
        else:
            self.runmode = runmode

        if control_mode is None:
            control_mode = 'Fractional modified'
        self.control_func = control_func_dict[control_mode]


        self.datacoll=datacoll
        self.plotflag = plotflag
        self.collect_freqs = collect_freqs
        
        if dt is None:
            self.dt = self.calc_dt(self.infection_rate,self.yc)
        
        if figures_folder is None:
            self.figures_folder = os.getcwd()+'/figures/'
            os.makedirs(self.figures_folder,exist_ok = True)
        else:
            self.figures_folder = figures_folder
        self.root_strain = Strain(0,population=self)
        self.max_mut=0
        self.t_intersection = np.inf
        self.n_contenders_for_intersection = 0


        self.root_strain.active_infected = self.initial_infected
        self.root_strain.susceptible_size = self.N - self.initial_infected
        
        self.ytot_history = []
        self.x_avg_history = []

        
        self.strain_couples = {}
        self.strain_set = [self.root_strain]
        self.alive_strains = [self.root_strain]
        self.changed_strains_flag=1
        self.update_distance_matrix()
        # self.datacollector = DataCollector(
        #     {0: lambda m: self.count_agents_by_strain(m, 0)}
        # )
        if not self.collect_freqs:
            self.datacollector = DataCollector(
                {
                "first_tb": lambda m: self.get_first_intersection_time_effect(m) [0],
                "first_d": lambda m: self.get_first_intersection_time_effect(m) [1],
                "t_x": lambda m: self.get_first_intersection_time_effect(m) [2],
                "chi_0": lambda m: self.get_first_intersection_time_effect(m) [3]}
            )
        else:
            self.datacollector = DataCollector(
                {
                "first_tb": lambda m: self.get_first_intersection_time_effect(m) [0],
                "first_d": lambda m: self.get_first_intersection_time_effect(m) [1],
                "t_x": lambda m: self.get_first_intersection_time_effect(m) [2],
                "freqs": lambda m:self.get_frequency_and_d(m)
                }
            )

        if seed is not None:
            self.seed = int(seed)
        else:
            seed = int(time.time() * 1000) % (2**32 - 1)
            self.seed = seed
        rng = np.random.Generator(np.random.PCG64(seed))
        self.rng = rng

    def get_distance_matrix(self):
        return np.array([[s1.tree_distance(s2) for s1 in self.alive_strains] for s2 in self.alive_strains])    
    
    def update_distance_matrix(self):
        if self.changed_strains_flag:
            self.distance_matrix = self.get_distance_matrix()

    def calc_x_avg(self):
        Is = np.array([float(s.active_infected) for s in self.alive_strains]).flatten()
        Ss = np.array([float(s.calc_susc_for_timestep()) for s in self.alive_strains]).flatten()
        
        return np.sum(Ss*Is)/np.sum(Is * self.N)
        
    def step(self):

        # internal_seed = self.rng.integers(0, 2**32 - 1)

        dt = self.dt
        
        Is = np.array([float(s.active_infected) for s in self.alive_strains]).flatten()
        Ss = np.array([float(s.calc_susc_for_timestep()) for s in self.alive_strains]).flatten()
        Rs = np.array([float(s.recovered_size) for s in self.alive_strains]).flatten()

        xs = Ss/self.N

        # x_avg = np.sum(Ss*Is)/np.sum(Is * self.N)
        x_root = self.root_strain.susceptible_size/self.N

        Itot = np.sum(Is) 
        ytot = Itot/self.N
        deltaI_infected = np.zeros(len(self.alive_strains))
        deltaI_recovered = np.zeros(len(self.alive_strains))

        # def calc_fitness(self, x_avg):
        # Itot = np.sum([x.active_infected for x in self.population.alive_strains])
        # R0 = self.population.infection_rate 
        # yc = self.population.yc
        # N= self.population.N
        
        # x= self.calc_susc_for_timestep()/self.population.N
        
        # f = R0 *self.population.control_func(Itot/N, x_avg, yc, R0)*x-1
        # self.fitness_history.append(f)

        R0_y = self.infection_rate*self.control_func(ytot, x_root, self.yc,self.infection_rate)
        try:
            deltaI_infected[Is>0] = self.rng.poisson(R0_y*xs *Is[Is>0] *dt).flatten()
        except IndexError:
            print('Is',Is.shape)

        deltaI_recovered[Is>0] = self.rng.poisson(self.recovery_rate*Is[Is>0]*dt) 
        deltaS_infections = - np.dot(np.exp(-self.distance_matrix/self.xi), deltaI_infected)

        mutations = np.zeros(len(self.alive_strains))
        mutations = self.rng.poisson(self.mutation_rate*Is[Is>0]*dt)
        
        new_strains = []
        for im, m in enumerate(mutations):
            if m>0:
                self.changed_strains_flag=1
                for i in range(m):
                    new_strains.append(self.alive_strains[im].mutate_random(self.max_mut+1,self.p_alpha,self))
                    self.max_mut+=1

        for i, strain in enumerate(self.alive_strains):
            strain.susceptible_size = np.maximum(strain.susceptible_size+ deltaS_infections[i],0)
            strain.active_infected = np.maximum(strain.active_infected+ deltaI_infected[i] - deltaI_recovered[i],0)
            strain.recovered_size = np.maximum(strain.recovered_size+ deltaI_recovered[i],0)
            try:
                if strain.active_infected == 0:
                    self.alive_strains.remove(strain)
                    self.changed_strains_flag=1
            except ValueError:
                print(strain.active_infected)
                print(deltaI_infected[i])
                pass

        self.alive_strains.extend(new_strains)
        self.strain_set.extend(new_strains)
        self.schedule.step()

        self.update_distance_matrix()

        self.check_intersection()
        self.datacollector.collect(self)

    def snapshot(self,t):
        x_avg = self.calc_x_avg()
        ytot = np.sum([s.active_infected for s in self.alive_strains])/self.N
        for strain in self.alive_strains:
            strain.calc_infected()
            strain.calc_susceptible()
            strain.calc_recovered()
            strain.calc_fitness(x_avg=x_avg)
            strain.calc_frequency()
            strain.times.append(t)
        for strain in self.alive_strains:
            strain.calc_selection_against_parent()
        self.x_avg_history.append(x_avg)
        self.ytot_history.append(ytot)

        
    def max_steps_func(self):
        if self.runmode=='end':
            dt =self.calc_dt(self.infection_rate,self.yc)

            T = self.calc_T(self.infection_rate,self.yc,self.N)
            n_steps = int(T/dt)
            return n_steps
        elif self.runmode=='peak':
            return self.peak_steps_func()
        else:
            raise ValueError("running mode unspecified")

    def get_final_frequencies(self):
        Itot = self.datacollector.model_vars['I'][-1]
        d_frequencies= []
        for strain in self.root_strain.children:
            d_frequencies.append((strain.distance_to_parent, strain.active_infected/Itot))
        return np.array(d_frequencies)

    def peak_steps_func(self):
        dt =self.calc_dt(self.infection_rate,self.yc)
        T = self.calc_t_peak(self.infection_rate,self.N)
        n_steps = int(T/dt)
        return n_steps
    
    def calc_t_peak(self,R0,N, k =None ):
        if k is not None:
            raise ValueError("kappa t peak not implemented!")
        
        I0 = self.initial_infected
        return np.log(1+ N/I0 *(R0-1)/R0*np.log(R0))/(R0-1)

    def find_x_inf(self,R0,yc):
        kappa = (R0-1)/yc
        if kappa>0:
            c = (R0-kappa+R0*kappa)/(kappa*(R0-kappa))
            rho = 1/(R0/kappa)
            y_x = lambda x: c*x**rho + x/(rho-1) - 1/kappa

            roots = spo.fsolve(y_x, 0)
            return roots[0]
        else:
            return np.real(-1/R0*sps.lambertw(-R0*np.exp(-R0)))

    def calc_dt(self,R0,yc):
        if yc==0:
            return .01*2.5/R0
        return .01*2.5*(np.log((R0-1)/yc) + 10)/R0

    def calc_T(self,R0,yc,N):
        return np.log(N)*(1/(R0-1) + 1/(1-R0*self.find_x_inf(R0,yc)))
    
    def calc_contenders_for_intersection(self, thresh = 1e-2):
        strains_with_history = [strain for strain in self.strain_set if len(strain.infected_history)>0]
        contenders = [strain for strain in strains_with_history if np.max(strain.frequency_history)>thresh]
        return len(contenders)

    def check_intersection(self):
        for strain in self.strain_set:
            strain.check_intersection()
            if strain.root_intersection_time<np.inf:
                self.t_intersection = min(self.t_intersection, strain.root_intersection_time)
                self.running=False
        
    def run_to_extinction(self, snapshot_interval = 10, ignore_running = False, max_iter= None, progress_bar = False):
        for i in tqdm(range(self.max_steps_func()), disable = not progress_bar):
            if len(self.alive_strains)==0:
                break
            if self.running==False and ignore_running==False:
                self.n_contenders_for_intersection = self.calc_contenders_for_intersection()
                break
            else:
                if self.root_strain.active_infected==0:
                    break
            if max_iter is not None and i>max_iter:
                self.n_contenders_for_intersection = self.calc_contenders_for_intersection()
                break
            self.step()
            
            if i%snapshot_interval==0:
                self.snapshot(i*self.dt)
        self.ilast = i

    def run_to_peak(self,snapshot_interval=10):
        for i in tqdm(range(self.peak_steps_func())):
            self.step()
            if i%snapshot_interval==0:
                self.snapshot(i*self.dt)
            self.snapshot(i*self.dt)
        self.ilast = i
        
    def plot_trajectories(self, save= True):
        R0 = self.infection_rate
        yc = self.yc
        N = self.N
        xi = self.xi
        mu = self.mutation_rate
        p_alpha = self.p_alpha
        
        ds = np.arange(0,1+np.max([strain.distance_to_parent for strain in self.strain_set]))
        color_ds= dict(zip(ds,plt.cm.jet(np.linspace(0,1,len(ds)))))
        
        fig,ax= plt.subplots(2,1,figsize=(15,15),sharex=True)
        plt.subplots_adjust(hspace=.1)
        y0p = (R0-1-np.log(R0))/R0

        xinf = self.find_x_inf(R0,yc)
        T = self.calc_T(R0,yc,N)
        zinf = 1-xinf


        legend_ds = {}
        for strain in self.strain_set:
            # print(strain)
            if strain.name==0:
                color='grey'
            else:
                color = color_ds[strain.distance_to_parent]

            I_i = np.array(strain.infected_history)
            if len(I_i)==0:
                continue


            S_i = np.array(strain.susceptible_history)
            d_pc = strain.distance_to_parent
            ax[1].plot(strain.times,I_i/N,color=color,)
            ax[0].plot(strain.times,S_i/N,color=color)


        # if kappa==0: ax[1].axvline((np.log( 1 + N*(1-1/R0)*np.log(R0)))/(R0-1),color='green')

        # ax[1].set_yscale('log')
        # ax[1].set_ylim(1/(2*N),10.1*y0p)
        ax[0].axhline(0,color='black',linestyle='--')
        # ax[1].axvline(self.calc_t_peak(self.infection_rate,self.N))
        
        # a/x[1].plot(np.arange(self.max_steps_func()*2)*self.dt,np.array(I_vals)/N)
        try:
            cmap = plt.cm.jet  # define the colormap
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[0] = (.5, .5, .5, 1.0)
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)
            bounds = ds/xi
            norm = mpl.colors.BoundaryNorm(ds/xi, cmap.N)
            ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=ax2, orientation='vertical', label=r'$d/\xi$', ticks=bounds[::2], boundaries=bounds[::2])#,format=FuncFormatter(fmt))#, format='%1f')
        except ValueError:
            pass
        ax[0].set_ylabel(r'$R_0 x_i(t)-1$, fitness')
        ax[1].set_ylabel(r'$y_i(t)$, inf. fraction')
        ax[1].set_xlabel(r'$t$, time [recovery periods]')
        fig.suptitle(fr'$R_0={R0}, y_c = {yc}, \mu N={mu*N}, \frac{{\langle d\rangle}}{{\xi}}={1/(p_alpha*xi):.2f} $',y=.95)

        counter=0
        replicates_folder = f'R0_{R0}_yc_{yc}_muN_{mu*N}_dxi_{1/(p_alpha*xi):.2f}/'
        os.makedirs(self.figures_folder+replicates_folder,exist_ok=True)
        
        figure_name = self.figures_folder+replicates_folder+f'replicate_{counter}.png'
        while os.path.exists(figure_name):
            counter+=1
            figure_name = self.figures_folder+replicates_folder+f'replicate_{counter}.png'
        if save: plt.savefig(figure_name,bbox_inches='tight')
        return ax
    
    @staticmethod
    def count_agents_by_status(model, status):
        return sum(1 for agent in model.schedule.agents if agent.state == status)

    @staticmethod
    def count_agents_by_strain(model, strain_name):
        return sum(1 for agent in model.schedule.agents if agent.strain is not None and agent.strain.name == strain_name )
    
    @staticmethod
    def get_first_intersection_time_effect(model):
        intersecters = sorted([strain for strain in model.strain_set if ~np.isinf(strain.root_intersection_time)], key = lambda x: x.root_intersection_time)
        try:
            first_intersecter= intersecters[0]
            return (first_intersecter.birth_time, first_intersecter.distance_to_root, first_intersecter.root_intersection_time, first_intersecter.initial_frequency)
        except IndexError:
            return (np.inf, np.inf, np.inf, np.inf)
    
    @staticmethod
    def get_frequency_and_d(model):
        model.snapshot(model.schedule.steps*model.dt)
        Itot = sum(1 for agent in model.schedule.agents if agent.state == 'I')
        d_frequencies= []
        for strain in model.strain_set:
            try:
                d_frequencies.append([strain.distance_to_parent, strain.active_infected/Itot])
            except ZeroDivisionError:
                pass
        return np.array(d_frequencies)
