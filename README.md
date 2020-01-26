# SpaceOpt: optimize discrete search spaces via predictive modeling

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

and if you can evaluate points from it, for example:

```python
spoint = {'a': 4, 'b': 0.0, 'c': 512, 'd': 'typeZ', 'e': False}
y = feval(spoint)
print(y)  # 0.123456
```

and if you want to find points that maximize or minimize evaluation objective, <b>in a better way than random search</b>, then use SpaceOpt:

```python
from spaceopt import SpaceOpt

spaceopt = SpaceOpt(search_space=search_space,
                    target_name='y',
                    objective='min')

for iteration in range(200):

    if iteration < 20:
        spoint = spaceopt.get_random()  # exploration
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

## License

MIT License (see [LICENSE](./LICENSE)).
