class Genome():
    def __init__(self, X_train, X_test, y_train, y_test, estimator, params):
        self._estimator = estimator
        self._params = params
        self._fitness = self.calc_fitness(X_train, X_test, y_train, y_test ,params)
    
    @property
    def estimator(self):
        return self._estimator

    @property
    def params(self):
        return self._params

    @property
    def fitness(self):
        return self._fitness

    @params.setter
    def params(self, new_params):
        self._fitness = self.calc_fitness(new_params)
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
        string = f'Genome = Params : {self.params}, Fitness : {self.fitness}'
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

    
    def calc_fitness(self, params, metric):
        """
        Parameters
        -------------------
        Takes in 1 genome/set of params
        """

        model = self.estimator

        model.set_params(params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results = self.evaluate(y_test, y_pred, metric)

        return results

    
    def mutate(self, param_grid, rate):
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
        print(f'original: {self}')

        length = len(self)
        choices = list(range(0, length))
        params = self.params.copy()
        keys = list(params.keys())

        ################################
        # Edge case 1: Only one choice, unable to mutate, so just remove from choices
        for index, (key, items) in enumerate(param_grid.items()):
            if len(items['range']) < 2:
                   choices.remove(index)
        ##################################

        for i in choices:
            prob = random.random()
            if prob < rate:   ## If 0 < prob and prob < rate, means falls under the possibility of mutating
                key = keys[i]
                target_val = params[key]
                items = param_grid[key]
                new_target_val = target_val

                while new_target_val == target_val:     ## Iterate until not the same
                    if items['dtype'] == 'category':
                        new_target_val = random.choice(items['range'])
                    elif items['dtype'] == 'int':
                        low, high = items['range']
                        new_target_val = random.randint(low, high)
                    elif items['dtype'] == 'float':
                        low, high = items['range']
                        new_target_val = round(random.uniform(low, high), 5)

                params[key] = new_target_val

                print(f"After mutation: '{key}': '{new_target_val}'")
            else:
                print(f'No mutation.')
                continue

        new_gene = Genome(self.estimator, params)

        return new_gene
    
    def evaluate(self, actual, predicted, metric):
        '''Returns desired metrics'''

        metrics = {
                    "r2":r2_score(actual, predicted), 
                    "rmse":mse_score(actual, predicted)**0.5,
                    "mse":mse_score(actual, predicted), 
                    "mae":mae_score(actual, predicted),
                    }
        return metrics[metric]

    
    class Population():
    def __init__(self, X_train, X_test, y_train, y_test, estimator, param_grid, pop_size, population = None):
        self._X_train = X_train
        self._X_test = X_test
        self._y_test = y_test
        self._y_train = y_train
        self._estimator = estimator
        self._param_grid = param_grid
        self._pop_size = pop_size
        if not population:  ## If no population
            self._population = self.generate_pop()
        else:
            self._population = population

    @property
    def X_train(self):
        return self._X_train
    
    @property
    def X_test(self):
        return self._X_test
    
    @property
    def y_train(self):
        return self._y_train
    
    @property
    def y_test(self):
        return self._y_test

    @property
    def estimator(self):
        return self._estimator
    
    @property
    def population(self):
        return tim_sort(self._population)
    
    @property
    def param_grid(self):
        return self._param_grid
    
    @property
    def pop_size(self):
        return self._pop_size

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
        return Population(self.param_grid, self.pop_size, pop_copy)
            
    
    def get_fit(self):
        """
        Helps to get list of fitness values
        """
        fits = []
        for i in self:
            fits += [i.fitness]
        return fits

    
    def calc_avg_fit(self):
        """
        For adaptive Mutation
        """
        fits = self.get_fit()
        return statistics.mean(fits)


    def generate_pop(self, X_train, X_test, y_train, y_test):
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
        pop = []
        for i in range(self.pop_size):
            params = {}
            for key, items in self.param_grid.items():
                if items['dtype'] == 'category':
                    params[key] = random.choice(items['range'])
                elif items['dtype'] == 'int':
                    low, high = items['range']
                    params[key] = random.randint(low, high)
                else:
                    low, high = items['range']
                    params[key] = round(random.uniform(low, high), 5)
            new_genome = Genome(X_train, X_test, y_train, y_test, self.estimator, params)
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


    def n_cross_over(self, genes, n, nco_rate = 0.5, verbose = True):
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
            print("p value = %.5f is more than %f. No cross over." %(prob, nco_rate))
            return [gene1, gene2]

        choices = list(range(1,len1-1))
        idx = list(np.random.choice(choices, size = n, replace = False))
        idxs = tim_sort(idx + [0, len1]) 
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
            
            # print(params33)
            # print(params44)
            
            params3 = merge_dicts(params3, params33)
            params4 = merge_dicts(params4, params44)
                
        gene3 = Genome(self.X_train, self.X_test, self.y_train, self.y_test, self.estimator, params3)
        gene4 = Genome(self.X_train, self.X_test, self.y_train, self.y_test, self.estimator, params4)

        if verbose:
            print(f'------------Parent Genomes------------\n{gene1}\n{gene2}\n')
            print(f'After {n}-point(s) crossover at index(es) : {idx}\n')
            print(f'------------Children Genomes-------------\n{gene3}\n{gene4}\n')

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
        fits = self.get_fit()[low:high]
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

    def mutation(self, type, inplace = True, **params):
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
        subset: [a, b) Genome at index a (inclusive) till b (exclusive)

        Output:
        Population with mutated genes, inplace

        Warning:
        Self.population has to be SORTED
        """
        f_max = max(self.get_fit())
        f_avg = self.calc_avg_fit()
        new_pop = []

        # if not subset:  
        #     subset = self.population
        # else: 
        #     a, b = subset
        #     subset = self.population[a:b]

        subset = self.population
        ## Iterate through all the genome
        for index, gene in enumerate(subset):
            if type == 'fitness':
                try:
                    k = params['k']
                except KeyError:
                    print("Using default k = (0.05, 0.06).")
                    k = (0.05,0.06)
                finally:
                    rate = self.fitness_mutation_rate(k, f_max, f_avg, f = gene.fitness)

            elif type == 'rank':
                r = index
                n = self.pop_size
                try:
                    p_max = params['p_max']
                except KeyError:
                    print("Using default p_max = 0.08.")
                    p_max = 0.08
                finally:
                    rate = self.rank_mutation_rate(p_max, r, n)

            elif type == 'random':
                try:
                    rate = params['rate']
                except KeyError:
                    print("Using default rate = 0.01.")
                    rate = 0.01

            else:
                raise ValueError("No such mutation type.")
            
            new_gene = gene.mutate(self.param_grid, rate)
            new_pop += [new_gene]
        
        if inplace:
            self.population = new_pop
            new_pop = self
        else:
            new_pop = Population(self.X_train, self.X_test, self.y_train, self.y_train, self.estimator, self.param_grid, self.pop_size, new_pop)

        return new_pop
    
    def best_solution(self):
        return self.population[0]

class GenomeGrid():
    def __init__(self, estimator, param_grid, max_evol, pop_size, mutation_type, scoring = 'mse', el_prop = 0.2, nco_rate = 0.5, **params):
        """'
        Parameters/Attributes
        -------------------
        estimator: Model
        param_grid: Parameters
        max_evol: Max evolution
        pop_size: Population size
        mutation_type: 'fitneses', 'random', 'rank'

        optional:
        scoring: Evaluation criteria
        elitism: Prop of population for elitism, default = 0.2
        cross_over_rate/nco_rate: Default = 0.5
        type: 'fitness', 'random', 'rank'
            Default
            - 'fitness' : k = (k1,k2) = (0.05,0.06)
            - 'random': rate = 0.5
            -'rank': p_max = 0.08
        
        """
        self._estimator = estimator
        self._param_grid = param_grid
        self._max_evol = max_evol
        self._pop_size = pop_size
        self._mutation_type = mutation_type
        self._scoring = scoring
        self._el_prop = el_prop
        self._nco_rate = nco_rate
        if mutation_type == 'fitness':
            self._mutation_para = params.get('k', (0.05, 0.06))
        elif mutation_type == 'random':
            self._mutation_para = params.get('rate', 0.5)
        elif mutation_type == 'rank':
            self._mutation_para = params.get('p_max', 0.08)
        else:
            raise Exception('No such mutation type')

    @property
    def estimator(self):
        return self._estimator

    @property
    def param_grid(self):
        return self._param_grid
    
    @property
    def max_evol(self):
        return self._max_evol
    
    @property
    def pop_size(self):
        return self._pop_size
    
    @property
    def mutation_type(self):
        return self._mutation_type
    
    @property
    def scoring(self):
        return self._scoring
    
    # @property
    # def k(self):
    #     if self._k:
    #         return self._k
    #     else:
    #         raise ValueError("k tuple not initialised")
        
    # @property
    # def rate(self):
    #     if self._rate:
    #         return self._rate
    #     else:
    #         raise ValueError("rate not intialised")
        
    # @property
    # def p_max(self):
    #     if self._p_max:
    #         return self._p_max
    #     else:
    #         raise ValueError("p_max not initialised")

    @property
    def mutation_para(self):
        return self._mutation_para    

    @property
    def nco_rate(self):
        return self._nco_rate

    @property
    def el_prop(self):
        return self._el_prop
    
    def train_model(self, X_train, X_test, y_train, y_test, verbose = True):
        pop = Population(X_train, X_test, y_train, y_test, self.estimator, self.param_grid, self._pop_size)
        for i in range(self.max_evol):
            
            ## Elitism: Choose top t
            top_t = pop.elitism(self.el_prop)
            parent_pairs = self.pop_size - len(top_t)

            ## Selection: Make a list of tuple pairs for crossover from subset
            parent_pairs = pop.selection(parent_pairs)

            ## Cross Over subset
            children = []
            for pair in parent_pairs:
                child = pop.n_cross_over(pair, 1, self.nco_rate, verbose)
                print(f'this is child {pair}: {child}')
                children += child
            
            pop = top_t + children
            pop = Population(self.estimator, self.param_grid, self._pop_size, pop)
            
            ## Mutate everything, including those carried over by elitism
            pop.mutation(self.mutation_type, self.mutation_para)          

            if verbose:
                print(f'--------------Generation {i}------------\n{pop}')
        
        best = pop.best_solution()

        return best
    
        

            




        

        


