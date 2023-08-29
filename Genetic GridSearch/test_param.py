from space import Categorical, Integer, Continuous
from sklearn_genetic import space

param_grid = {'kernel': Categorical(['linear','poly']),
                        'C': Continuous(0,1),
                        'gamma': Continuous(0,1)}

param_grid_ga = {'kernel': space.Categorical(['linear','poly']),
                        'C': space.Continuous(0,1),
                        'gamma': space.Continuous(0,1)}

param_grid_search = {'kernel': ['linear','poly'],
                     'C': [random.uniform(0,1) for i in range(10)],
                     'gamma': [random.uniform(0,1) for i in range(10)]}