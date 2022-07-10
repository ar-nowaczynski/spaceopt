# SpaceOpt: hyperparameter optimization via gradient boosting regression

[![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/spaceopt?color=1)](https://pypi.org/project/spaceopt/)
[![license](https://img.shields.io/pypi/l/spaceopt)](https://github.com/ar-nowaczynski/spaceopt)

SpaceOpt is a hyperparameter optimization algorithm that uses gradient boosting regression to find the most promising candidates for the next trial by predicting their evaluation score.

## Installation

```bash
$ pip install spaceopt
```

## Usage

1. Define a discrete hyperparameter search space, for example:

```python
search_space = {
    'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # list of ordered numbers: ints
    'b': [-3.5, -0.1, 0.0, 2.5, 10.0],    # list of ordered numbers: floats
    'c': [256, 512, 1024, 2048],          # another list of ordered numbers
    'd': ['ABC', 'IJK', 'XYZ'],           # categorical variable
    'e': [True, False],                   # boolean variable
    # ... (add as many as you need)
}
```

2. Define your evaluation function:

```python
def evaluation_function(spoint: dict) -> float:
    # your code (e.g. model fit)
    return y  # score (e.g. model accuracy)

spoint = {'a': 4, 'b': 0.0, 'c': 512, 'd': 'XYZ', 'e': False}
y = evaluation_function(spoint)
```

3. Use SpaceOpt for a hyperparameter optimization:

```python
from spaceopt import SpaceOpt

spaceopt = SpaceOpt(search_space=search_space,
                    target_name='y',
                    objective='maximize')  # or 'minimize'

for iteration in range(200):
    if iteration < 20:
        spoint = spaceopt.get_random()     # exploration
    else:
        spoint = spaceopt.fit_predict()    # exploitation

    spoint['y'] = evaluation_function(spoint)
    spaceopt.append_evaluated_spoint(spoint)
```

More examples [here](https://github.com/ar-nowaczynski/spaceopt/tree/master/examples).

## Advanced

- get multiple points by setting `num_spoints`:
```python
spoints = spaceopt.get_random(num_spoints=2)
# or
spoints = spaceopt.fit_predict(num_spoints=5)
```

- control exploration vs. exploitation behaviour by adjusting `sample_size` (default=10000), which is the number of candidates sampled for ranking:
```python
spoint = spaceopt.fit_predict(sample_size=1000)  # decreasing `sample_size` increses exploration
spoint = spaceopt.fit_predict(sample_size=100000)  # increasing `sample_size` increses exploitation
```

- add manually selected evaluation points to SpaceOpt:
```python
my_spoint = {'a': 8, 'b': -3.5, 'c': 256, 'd': 'IJK', 'e': False}
my_spoint['y'] = evaluation_function(my_spoint)
spaceopt.append_evaluated_spoint(my_spoint)
```

- learn more by reading the code, there are only 3 classes: [SpaceOpt](https://github.com/ar-nowaczynski/spaceopt/blob/master/spaceopt/optimizer.py), [Space](https://github.com/ar-nowaczynski/spaceopt/blob/master/spaceopt/space.py) and [Variable](https://github.com/ar-nowaczynski/spaceopt/blob/master/spaceopt/variable.py).

## License

MIT License (see [LICENSE](https://github.com/ar-nowaczynski/spaceopt/blob/master/LICENSE)).
