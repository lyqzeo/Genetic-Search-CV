import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Tuple
from numpy import random
import itertools
import math
import statistics
from tabulate import tabulate
import time
import joblib
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import mean_absolute_percentage_error as mape_score
from sklearn_genetic import GASearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split
# from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

import optuna
import plotly.graph_objects as go

from space import Space, Categorical, Integer, Continuous
from helper import *
from evaluate import Evaluate

class Genome():
    def __init__(self, hyperparameters, **params):

        self.initialise_shared_var(**params)

        self.params = hyperparameters
        self.fitness = self.calc_fitness(self.params)

    @classmethod
    def initialise_shared_var(cls, **params):
        try:
            cls.X_train = params['X_train']
            cls.y_train = params['y_train']
            cls.estimator = params['estimator']
            cls.metric = params['metric']
            cls.cv = params['cv']
            cls.cv_shuffle = params['cv_shuffle']
        except KeyError:
            return

        
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self.fitness = self.calc_fitness(new_params)
        self._params = new_params


    ## Comparing operators
    def __lt__(self, other):
        # Define custom behavior for "<"
        return self.fitness < other.fitness

    def __le__(self, other):
        # Define custom behavior for "<="
        return self.fitness <= other.fitness

    def __ge__(self, other):
        # Define custom behavior for ">="
        return self.fitness >= other.fitness

    def __eq__(self, other):
        # Define custom behavior for "=="
        return self.fitness == other.fitness

    def __ne__(self, other):
        # Define custom behavior for "!="
        return self.fitness != other.fitness
    
    def __len__(self):
        return len(self.params.keys())
    
    def __repr__(self):
        string = f'(Genome = Params : {self.params}, Fitness : {self.fitness})'
        return string
    
    def extract(self, start, stop = None, step = 1):
        """
        Just for ease of slicing
        
        Parameters
        -----------------
        key: Can be slice object
        """

        if not stop: ## If stop == None, means only want extract one
            return dict([list(self.params.items())[start]])
            
        elif stop > start and stop <= len(self):
            
            target = dict([list(self.params.items())[i] for i in range(start, stop, step)])
            return target
        else:
            raise IndexError(f"Index out of range.")

    
    def calc_fitness(self, params):
        """
        Parameters
        -------------------
        Takes in 1 genome/set of params
        """
        results = []

        cv_shuffle = self.cv if self.cv_shuffle else 1

        kf = RepeatedKFold(n_splits = self.cv, n_repeats = cv_shuffle, random_state = 2)

        for train, test in kf.split(self.X_train):
            X_train, X_test, y_train, y_test = self.X_train.iloc[train], self.X_train.iloc[test], self.y_train.iloc[train], self.y_train.iloc[test]

            model = clone(self.estimator)
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            result = Evaluate()(y_test, y_pred, self.metric)
            model = None
            results += [result]

        return statistics.mean(results)

    
    def mutate(self, param_grid, rate, verbose = False):
        """"
        Mutate one gene

        Parameters
        --------------------
        choices: Param_grid for choosing
        rate: Mutation rate/ Probability of one parameter undergoing mutation

        Returns
        -------------------
        Returns new gene
        """
        if verbose:
            print(f'original: {self}')

        length = len(self)
        choices = list(range(0, length))
        params = self.params.copy()
        keys = list(params.keys())

        for i in choices:
            prob = random.random()
            if prob < rate:   ## If 0 < prob and prob < rate, means falls under the possibility of mutating
                key = keys[i]
                target_val = params[key]
                items = param_grid[key]
                new_target_val = target_val

                if isinstance(items, Categorical):
                    if len(items) == 1:
                        continue
                else:
                    while new_target_val == target_val:     ## Iterate until not the same 
                        new_target_val = items.sample()     ## Ok but i'm not sure if i should do this because sklearn didnt and it
                                                            ## kind of ruins the sampling from a distribution purose

                params[key] = new_target_val
                if verbose:
                    print(f"After mutation: '{key}': '{new_target_val}'")
            else:
                if verbose:
                    print(f'No mutation.')
                continue

        new_gene = Genome(params)

        return new_gene
    


class Population():


    def __init__(self, population = None, **params):

        self.initialise_shared_var(**params)

        if not population:  ## If no population
            self._population = self.generate_pop()
        else:
            self._population = population

    @classmethod
    def initialise_shared_var(cls, **params):
        
        try:
            cls.X_train = params['X_train']
            cls.y_train = params['y_train']
            cls.estimator = params['estimator']
            cls.param_grid = params['param_grid']
            cls.pop_size = params['pop_size']
            cls.metric = params['metric']
            cls.cv = params['cv']
            cls.cv_shuffle = params['cv_shuffle']
            cls.direction = params['direction']
        except KeyError:
            return

    @property
    def population(self):
        return tim_sort(self._population, self.direction)
    
    @property
    def get_fit(self):
        """
        Helps to get list of fitness values
        """
        fits = []
        for i in self.population:
            fits += [i.fitness]
        return fits


    @population.setter
    def population(self, new_pop):
        self._population = new_pop

    ## Used this only for sorting so it's easier
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.population[key]
        
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.n)
            target = [self.population[i] for i in range(start, stop, step)]
            return target
        else:
            raise TypeError(f"Invalid Argument type. Must be int or slice.")
        
    def __repr__(self):
        string = ""
        for genome in self.population:
            string += f'{genome}\n'
        return string
        
    def copy(self):
        """
        Makes a copy of the current generation
        """
        pop_copy = [genome for genome in self.population]
        return Population(pop_copy)
            
    
    def calc_avg_fit(self):
        """
        For adaptive Mutation
        """
        fits = self.get_fit
        return statistics.mean(fits)


    def generate_pop(self):
        random.seed(3)
        """
        To initialize population

        Parameters
        --------------
        n: Number of Genomes

        Returns
        ---------------
        None

        """
        genome_attributes = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'estimator': self.estimator,
            'metric': self.metric,
            'cv': self.cv,
            'cv_shuffle': self.cv_shuffle
        }
        
        pop = []
        for i in range(self.pop_size):
            params = {}
            for key in self.param_grid.parameters:
                params[key] = self.param_grid[key].sample()
            new_genome = Genome(params, **genome_attributes)
            pop += [new_genome]
        return pop


    def elitism(self, n):
        """
        To choose which n Genomes to bring over to next gen

        Parameters
        ------------------
        n: prop of Genomes, default 0.2 of population

        Returns
        -----------------
        Top n Genomes
        """
        if not 0 < n < 1:
            raise ValueError("n must be between 0 and 1")
        n = round(n*self.pop_size)
        top_n = self.population[0:n]

        return top_n


    def n_cross_over(self, genes, n, nco_rate = 0.5, verbose = False):
        """
        To cross over 2 genomes
        
        Parameters
        ------------------------
        genes: (a,b) Cross over genomes in pop[a] and pop[b]
        n: Number of points to cross over
        verbose: Illustrate parents to genomes
        nco_rate: Cross over rate/ Probability of genes undergoing crossover
            -> If rate = 0: None of the genes can undergo mutation
            -> If rate = 1: All of the genes can undergo mutation
        
        Returns
        ----------------------
        2 Genomes, Crossed over, not in place
        """
        a,b = genes
        a = int(a)
        b = int(b)
        gene1 = self.population[a]
        gene2 = self.population[b]

        len1 = len(gene1)
        len2 = len(gene2)
        if len1 != len2:
            raise ValueError("Genomes must be of same length")
        
        if len1 < 2:
            raise Exception("Too short!")
        
        if n >= len1 -1:
            raise Exception(f"Too much cross-over points. Maximum is {len1-1}.")

        prob = random.random()

        if prob > nco_rate:  
            if verbose:
                print("p value = %.5f is more than %f. No cross over." %(prob, nco_rate))
            return [gene1, gene2]

        choices = list(range(1,len1))
        idx = list(np.random.choice(choices, size = n, replace = False))
        idxs = tim_sort(idx + [0, len1], dir = 0) 
        sections = n+1
        params3 = {}
        params4 = {}

        for i in range(sections):
            ## Even
            if i%2 == 0:
                params33 = gene1.extract(idxs[i], idxs[i+1])
                params44 = gene2.extract(idxs[i], idxs[i+1])
                
                    
            ## Odd
            else:
                params33 = gene2.extract(idxs[i], idxs[i+1])
                params44 = gene1.extract(idxs[i], idxs[i+1])
            
            params3 = merge_dicts(params3, params33)
            params4 = merge_dicts(params4, params44)
                
        gene3 = Genome(params3)
        gene4 = Genome(params4)

        if verbose:
            print(f'------------Parent Genomes------------\n{gene1}\n{gene2}\n')
            print(f'After {n}-point(s) crossover at index(es) : {idx}\n')
            print(f'------------Children Genomes-------------\n{gene3}\n{gene4}\n')
            print('================================================')

        return [gene3, gene4]


    def selection(self, n, subset = None):
        """"
        Purpose:
        Helps n_cross_over by choosing what 2 genomes to put in
        
        Input:
        'n': Number of parent pairs
        'subset': [low,high) Genes in population to select from eg. gene index 4:7 (exclusive)

        Output:
        List of tuples of indexes: [ (parent1,parent2) ,(parent3, parent4)]
        """
        if not subset:
            low, high = (0, self.pop_size)
            size = self.pop_size
        else:
            low, high = subset
            size = high -low 

        max = math.comb(size, 2)
        if n > max:
            raise ValueError(f'Too much combinations. Maximum is {max}')

        pairs = []
        choices = list(range(low, high))
        fits = self.get_fit[low:high]
        p = [i/sum(fits) for i in fits]

        for i in range(n):
            parent1, parent2 = random.choice(choices, p = p, size = 2, replace = False)
            pairs += [(parent1, parent2)]
        
        return pairs
    
    def fitness_mutation_rate(self, k, f_max, f_avg, f):
        """
        Purpose:
        Fitness-based adaptive mutation

        Input:
        k: (k1,k2) Tuple of 2 for constant, k1,k2 in (0,1)
        f_max: Maximum fitness of population
        f_avg: Average fitness of population
        f: Current fitness value of Genome
        
        Output:
        rate -> int
        """
        k1, k2 = k
        # print(f'k: {k}, f_max; {f_max}, f_avg: {f_avg}, f: {f}')
        
        if f >= f_avg:      ## High quality solution
            rate = k1*( (f_max - f)/(f_max-f_avg) )
        else:               ## Low quality solution
            rate = k2
        return rate

    def rank_mutation_rate(self, p_max, r, n):
        """"
        Purpose:
        Rank-based adaptive mutation
        
        Input:
        p_max: Maximum mutation probability
        r: Rank of chromosome
        n: population size

        Output:
        rate -> int
        """
        p = p_max*( 1- (r-1)/(n-1))
        return p

    def mutation(self, type, params, verbose = False, inplace = True):
        """
        Purpose:
        Randomly select from population without replacement and mutate

        Input:
        type: 'fitneses', 'random', 'rank'
            Default
            - 'fitness' : k = (k1,k2) = (0.05,0.06)
            - 'random': rate = 0.5
            -'rank': p_max = 0.08
        inplace: Mutate on spot

        Output:
        Population with mutated genes, inplace

        Warning:
        Self.population has to be SORTED
        """
        f_max = max(self.get_fit)
        f_avg = self.calc_avg_fit()
        new_pop = []

        subset = self.population
        ## Iterate through all the genome
        for index, gene in enumerate(subset):
            if type == 'fitness':
                k = params['k']
                rate = self.fitness_mutation_rate(k, f_max, f_avg, f = gene.fitness)

            elif type == 'rank':
                r = index
                n = self.pop_size
                p_max = params['p_max']
                rate = self.rank_mutation_rate(p_max, r, n)

            elif type == 'random':
                rate = params['rate']

            else:
                raise ValueError("No such mutation type.")
            
            new_gene = gene.mutate(self.param_grid, rate)
            new_pop += [new_gene]
        
        if inplace:
            self.population = new_pop
            new_pop = self
        else:
            new_pop = Population(new_pop)

        return new_pop
    
    def best_solution(self):
        return self.population[0]

class GenomeGrid():
    """
    Parameters
    -------------------
    est: Model
    param_grid: Parameters
    max_evol: Max evolution
    pop_size: Population size
    mutation_type: 'fitneses', 'random', 'rank'

    optional:
    cross_valid
    scoring: Evaluation criteria -> need to change!!!
    elitism: Prop of population for elitism, default = 0.2
    cross_over_rate/nco_rate: Default = 0.5
    type: 'fitness', 'random', 'rank'
        Default
        - 'fitness' : k = (k1,k2) = (0.05,0.06)
        - 'random': rate = 0.5
        -'rank': p_max = 0.08
    direction: "maximize", "minimize"

    Attributes
    ----------------------
    X_train, y_train, X_test, y_test
    best_hyperparameters
    problem_type: Regression/Classification

    
    """


    def __init__(
            self, 
            X, 
            y, 
            est, 
            parameters_grid, 
            cv = 5, 
            cv_shuffle = False, 
            max_evol = 100, 
            pop_size = 10, 
            mutation_type = 'rank', 
            scoring = 'recall', 
            el_prop = 0.2, 
            nco_rate = 0.5, 
            direction = "maximize",
            **params):

        self.estimator = est
        self.param_grid = Space(parameters_grid)
        self.max_evol = max_evol
        self.pop_size = pop_size
        self.mutation_type = mutation_type
        self.metric = scoring
        self.el_prop = el_prop
        self.nco_rate = nco_rate
        self.best_hyperparameters = None
        self.cv = cv
        self.cv_shuffle = cv_shuffle
        self.direction = 1 if direction == "maximize" else 0
        
        
        ## For evaluation function
        Evaluate.initialise(y = y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 123)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if mutation_type == 'fitness':
            self.mutation_para = {'k': params.get('k', (0.05, 0.06))}
        elif mutation_type == 'random':
            self.mutation_para = {'rate': params.get('rate', 0.5)}
        elif mutation_type == 'rank':
            self.mutation_para = {'p_max': params.get('p_max', 0.08)}
        else:
            raise Exception('No such mutation type')

    @property
    def best_model(self):
        best_est = joblib.load('best_estimator.pkl')
        return best_est


    @property
    def history(self):
        history_log = joblib.load('history.pkl')
        return history_log


    def fit(self, early_term = True, early_term_thresh = 0.0001, log = True, refit = True, verbose = False):
        """
        Main function to fit
        
        Parameters
        ------------------------
        early_term: terminatate algorithms early, when flag >= 10
        early_term_threshold: Flag += 1
        refit: Auto fit to best model
        verbose: True will also activate log (Write history in the file)


        """
        ## For early term: flag += 1 when change <= thresh
        if early_term:
            flag = 0
        
        
        start = time.time()
        
        population_attributes = {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "estimator": self.estimator,
            "param_grid": self.param_grid,
            "pop_size": self.pop_size,
            "metric": self.metric,
            "cv": self.cv,
            "cv_shuffle": self.cv_shuffle,
            "direction": self.direction
        }
        
        pop = Population(**population_attributes)

        ## For history
        gen = []
        fitness = []
        fitnessMin = []
        fitnessMax = []

        ## For logging
        if verbose:
            string = "Gen\t\tFitness\t\tFitnessMin\t\tFitnessMax\n"
            with open(log_file, 'w') as log:
                log.write(string)
            print(string)
            

        for i in range(self.max_evol):

            ## Early termination
            if early_term:
                if flag >= 10:
                    break

            ## Elitism: Choose top t
            top_t = pop.elitism(self.el_prop)
            parent_pairs = self.pop_size - len(top_t)

            ## Selection: Make a list of tuple pairs for crossover from subset
            parent_pairs = pop.selection(parent_pairs)

            ## Cross Over subset
            children = []
            for pair in parent_pairs:
                child = pop.n_cross_over(pair, 1, self.nco_rate, verbose)
                children += child
            
            pop = top_t + children
            pop = Population(pop)
            
            ## Mutate everything, including those carried over by elitism
            pop.mutation(self.mutation_type, self.mutation_para, verbose = verbose)

            ## For log/History
            mini = pop.get_fit[0]
            maxi = pop.get_fit[-1]
            avg = pop.calc_avg_fit()
            
            gen.append(i)
            fitness.append(avg)
            fitnessMin.append(mini)
            fitnessMax.append(maxi)

            if early_term:
                if i >= 1:
                    if abs( avg - fitness[-1] ) < early_term_thresh:
                        flag += 1
                    else:
                        flag = 0

        
            if verbose:
                string = f'{i}\t{avg}\t{mini}\t{maxi}\n'
                with open(log_file, "a") as log:
                    log.write(string)
                print(string)



        best = pop.best_solution()
        self.best_hyperparameters = best.params
        history = {'gen':gen, 'fitness':fitness, 'FitnessMin': fitnessMin, 'fitnessMax': fitnessMax}
        joblib.dump(history, 'history.pkl')

        if refit:
            model = clone(self.estimator)
            model.set_params(**self.best_hyperparameters)
            model.fit(self.X_train, self.y_train)
            joblib.dump(model, 'best_estimator.pkl')

        end = time.time()
        self.time_taken = end-start
        
        return best



    def compare(self, param_grid_search, param_grid_ga, plot = True):
        if not self.time_taken:
            raise Exception('Model not fitted yet!')
        
        pred = self.best_model.predict(self.X_test)
        cur_result = round(Evaluate()(self.y_test, pred, self.metric),5)

        # print(classification_report(self.y_test, pred))

        
        scoring = make_scorer(Evaluate(), metric = self.metric, greater_is_better = bool(self.direction))

        ## sklearn's Genetic Algo
        start = time.time()
        model_ga = clone(self.estimator)
        model_ga = GASearchCV(estimator = model_ga, param_grid = param_grid_ga, verbose = False, scoring = scoring, population_size = self.pop_size, generations = self.max_evol-1)
        model_ga.fit(self.X_train, self.y_train)
        end= time.time()
        time_ga = end - start

        pred = model_ga.predict(self.X_test)
        ga_result = round(Evaluate()(self.y_test, pred, self.metric),5)

        # print(classification_report(self.y_test, pred))


        ## sklearn's GridSearch
        start = time.time()
        model_gs = clone(self.estimator)
        model_gs = GridSearchCV(estimator = model_gs, param_grid = param_grid_search, scoring = scoring , return_train_score = True)
        model_gs.fit(self.X_train, self.y_train)
        end = time.time()
        time_gs = end - start
        
        pred = model_gs.predict(self.X_test)
        gs_result = Evaluate()(self.y_test, pred, self.metric)

        # print(classification_report(self.y_test, pred))

        ## optuna
        def op(trial):
            params = {
                'kernel': trial.suggest_categorical('kernel', ['linear','poly']),
                'C': trial.suggest_float('C', 0, 1),
                'gamma': trial.suggest_float('gamma', 0, 1)
            }

            model = clone(self.estimator)
            model.set_params(**params)
            model.fit(self.X_train,self.y_train)
            pred = model.predict(self.X_test)
            result = round(Evaluate()(self.y_test, pred, self.metric),5)

            return result

        start = time.time()
        optuna.logging.disable_default_handler()
        direction = "maximize" if self.direction == 1 else "minimize"
        model_op = optuna.create_study(direction = direction)
        model_op.optimize(op, n_trials = self.max_evol)
        end = time.time()
        time_op = end - start

        model_optuna = clone(self.estimator)
        model_optuna.set_params(**model_op.best_trial.params)
        model_optuna.fit(self.X_train, self.y_train)
        pred = model_optuna.predict(self.X_test)
        op_result = round(Evaluate()(self.y_test, pred, self.metric),5)

        # print(classification_report(self.y_test, pred))




        table = [["         ", "My GASearch", "sklearn GASearch", "sklearn GridSearchCV", "Optuna"],
                 ["Test " + self.metric, cur_result, ga_result, gs_result, op_result],
                 ["Time Taken", self.time_taken, time_ga, time_gs, time_op]]

        print(tabulate(table, headers = "firstrow", tablefmt="github"))

        print(f'My GASearch: {self.best_hyperparameters}\nSklearn GASearchCV: {model_ga.best_params_}\nSklearn GridSearchCV: {model_gs.best_params_}\nOptuna: {model_op.best_params}\n')

        if plot:
            y1 = self.history['fitness']
            y2 = [ abs(i) for i in model_ga.history['fitness'] ]
            y3 = [abs(i) for i in list(model_gs.cv_results_['mean_train_score'])]
            trials = model_op.get_trials()
            y4 = [trial.value for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]


            x = list(range(self.max_evol))
            n = max(len(y2), len(y3))

            fig = go.Figure()

            fig.add_trace(go.Scatter(x = x[:len(y1)], y = y1, name = "My GASearch", mode = 'lines'))
            fig.add_trace(go.Scatter(x = x, y = y2, name = "Sklearn GASearchCV", mode = 'lines'))
            fig.add_trace(go.Scatter(x = x[:n], y = y3[:n], name = "Sklearn GridSearchCV", mode = 'lines'))
            fig.add_trace(go.Scatter(x = x[:n], y = y4, name = "Optuna", mode = 'lines'))

            fig.show()
            

        return




