# SpaceOpt: optimize discrete search space via gradient boosting regression
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI version](https://img.shields.io/pypi/v/spaceopt?color=1)](https://pypi.org/project/spaceopt/)
[![license](https://img.shields.io/pypi/l/spaceopt)](https://github.com/ar-nowaczynski/spaceopt)


SpaceOpt is an optimization algorithm for discrete search spaces, that uses gradient boosting regression to find the most promising points for the evaluation by predicting the evaluation score. Training data is gathered on the fly with the preference to perform random or human-guided exploration at the beginning.

## Usage

If you have discrete search space - for example:

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
y = your_evaluation_function(spoint)
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

    spoint['y'] = feval(spoint)
    spaceopt.append_evaluated_spoint(spoint)
```

More examples [here](./examples/).

## Installation

```
$ pip install spaceopt
```

## Advanced

- get multiple points by `num_spoints`:
```
spoint_list = spaceopt.get_random(num_spoints=2)
# or
spoint_list = spaceopt.fit_predict(num_spoints=5)
```

- control exploitation behaviour by adjusting `sample_size`, which is the number of unevaluated points sampled for ranking (when you decrease`sample_size` exploration increses):
```
spoint = spaceopt.fit_predict(sample_size=100)
```

- add your own evaluation points to SpaceOpt:
```
my_spoint = {'a': 8, 'b': -4.4, 'c': 256, 'd': 'typeY', 'e': False}
my_spoint['y'] = feval(my_spoint)
spaceopt.append_evaluated_spoint(my_spoint)
```

- learn more by reading the code, there are only 3 classes: [SpaceOpt](spaceopt/optimizer.py), [Space](spaceopt/space.py) and [Variable](spaceopt/variable.py).

## License

MIT License (see [LICENSE](./LICENSE)).
