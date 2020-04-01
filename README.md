# SpaceOpt: optimize discrete search space via gradient boosting regression

[![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/spaceopt?color=1)](https://pypi.org/project/spaceopt/)
[![license](https://img.shields.io/pypi/l/spaceopt)](https://github.com/ar-nowaczynski/spaceopt)

SpaceOpt is an optimization algorithm for discrete search spaces that uses gradient boosting regression to find the most promising candidates for evaluation by predicting their evaluation score. Training data is gathered sequentially and random or human-guided exploration can be easily incorporated at any stage.

## Usage

If you have discrete search space, for example:

```python
search_space = {
    'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # list of ordered numbers: ints
    'b': [-4.4, -2.5, -1.5, 0.0, 3.7],    # list of ordered numbers: floats
    'c': [128, 256, 512, 1024],           # another list of ordered numbers
    'd': ['typeX', 'typeY', 'typeZ'],     # categorical variable
    'e': [True, False],                   # boolean variable
    # ... (add as many as you need)
}
```

and if you can evaluate points from it:

```python
spoint = {'a': 4, 'b': 0.0, 'c': 512, 'd': 'typeZ', 'e': False}
y = evaluation_function(spoint)
print(y)  # 0.123456
```

and if you want to find points that maximize or minimize your evaluation function, <b>in a better way than random search</b>, then use SpaceOpt:

```python
from spaceopt import SpaceOpt

spaceopt = SpaceOpt(search_space=search_space,
                    target_name='y',
                    objective='min')     # or 'max'

for iteration in range(200):

    if iteration < 20:
        spoint = spaceopt.get_random()   # exploration
    else:
        spoint = spaceopt.fit_predict()  # exploitation

    spoint['y'] = evaluation_function(spoint)
    spaceopt.append_evaluated_spoint(spoint)
```

More examples [here](https://github.com/ar-nowaczynski/spaceopt/tree/master/examples).

## Installation

```bash
$ pip install spaceopt
```

## Advanced

- get multiple points by setting `num_spoints`:
```python
spoint_list = spaceopt.get_random(num_spoints=2)
# or
spoint_list = spaceopt.fit_predict(num_spoints=5)
```

- control exploitation behaviour by adjusting `sample_size` (default=10000), which is the number of candidates sampled for ranking (decreasing `sample_size` increses exploration):
```python
spoint = spaceopt.fit_predict(sample_size=100)
```

- add your own evaluation points to SpaceOpt:
```python
my_spoint = {'a': 8, 'b': -4.4, 'c': 256, 'd': 'typeY', 'e': False}
my_spoint['y'] = evaluation_function(my_spoint)
spaceopt.append_evaluated_spoint(my_spoint)
```

- be creative about how to use SpaceOpt;

- learn more by reading the code, there are only 3 classes: [SpaceOpt](https://github.com/ar-nowaczynski/spaceopt/blob/master/spaceopt/optimizer.py), [Space](https://github.com/ar-nowaczynski/spaceopt/blob/master/spaceopt/space.py) and [Variable](https://github.com/ar-nowaczynski/spaceopt/blob/master/spaceopt/variable.py).

## License

MIT License (see [LICENSE](https://github.com/ar-nowaczynski/spaceopt/blob/master/LICENSE)).
