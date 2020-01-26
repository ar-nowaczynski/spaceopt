import random
from spaceopt import SpaceOpt
random.seed(123456)


def search_space():
    return {
        'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'b': [-5.5, -4.4, -3.3, -2.2, -1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5],
        'c': [128, 256, 512, 1024],
        'd': ['typeX', 'typeY', 'typeZ'],
        'e': [True, False],
        'f': [10000],
    }


def test_SpaceOpt__init__():
    SpaceOpt(search_space=search_space(), target_name='y', objective='min')
    SpaceOpt(search_space=search_space(), target_name='y', objective='max')
    SpaceOpt(search_space=search_space(), target_name='score', objective='maximize')
    SpaceOpt(search_space=search_space(), target_name='max', objective='minimize')

    try:
        SpaceOpt(search_space=search_space(), target_name=1, objective='min')
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "target_name is of type <class 'int'>, but it should be of type <class 'str'>."

    try:
        SpaceOpt(search_space=search_space(), target_name=None, objective='min')
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "target_name is of type <class 'NoneType'>, but it should be of type <class 'str'>."

    try:
        SpaceOpt(search_space=search_space(), target_name='', objective='min')
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "target_name is empty."

    try:
        SpaceOpt(search_space=search_space(), target_name='y', objective='maxi')
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "objective should be one of: ('maximize', 'minimize', 'max', 'min')."

    try:
        SpaceOpt(search_space=search_space(), target_name='b', objective='min')
    except Exception as e:
        assert isinstance(e, RuntimeError)
        assert str(e) == "target_name='b' should not be in search space variables: ['a', 'b', 'c', 'd', 'e', 'f']."


def test_SpaceOpt_append_evaluated_spoint():
    spaceopt = SpaceOpt(search_space=search_space(), target_name='y', objective='min')

    try:
        spaceopt.append_evaluated_spoint(['a', 'b', 'c', 'd', 'e'])
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "evaluated_spoint is of type <class 'list'>, but it should be of type <class 'dict'>."

    try:
        spaceopt.append_evaluated_spoint({'a': 16, 'b': 3.3, 'c': 512, 'd': 'typeX', 'e': False, 'f': 10000})
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "spoint={'a': 16, 'b': 3.3, 'c': 512, 'd': 'typeX', 'e': False, 'f': 10000} is not evaluated, target_name='y' is not found."

    try:
        spaceopt.append_evaluated_spoint({'a': 16, 'b': 3.3, 'c': 512, 'd': 'typeX', 'e': False, 'f': 10000, 'y': 1})
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "evaluated_spoint has 'y' with value=1 of type <class 'int'>, but it should be of type <class 'float'>."

    spaceopt.append_evaluated_spoint({'a': 16, 'b': 3.3, 'c': 512, 'd': 'typeX', 'e': False, 'f': 10000, 'y': 1.0})


def test_SpaceOpt_get_random():
    spaceopt = SpaceOpt(search_space=search_space(), target_name='y', objective='min')

    try:
        spaceopt.get_random(num_spoints=1.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "num_spoints is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt.get_random(num_spoints=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "num_spoints should be greater than 0."

    try:
        spaceopt.get_random(sample_size=100.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "sample_size is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt.get_random(sample_size=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "sample_size should be greater than 0."

    spoints = spaceopt.get_random(num_spoints=5, sample_size=1000)
    assert isinstance(spoints, list)
    assert len(spoints) == 5

    spoint = spaceopt.get_random(num_spoints=1, sample_size=10)
    assert isinstance(spoint, dict)
    assert set(spoint.keys()) == set(spoints[-1].keys())


def test_SpaceOpt_fit_predict():
    spaceopt = SpaceOpt(search_space=search_space(), target_name='y', objective='min')

    try:
        spaceopt.fit_predict(num_spoints=2.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "num_spoints is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt.fit_predict(num_spoints=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "num_spoints should be greater than 0."

    try:
        spaceopt.fit_predict(num_boost_round=2048.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "num_boost_round is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt.fit_predict(num_boost_round=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "num_boost_round should be greater than 0."

    try:
        spaceopt.fit_predict(sample_size=100.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "sample_size is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt.fit_predict(sample_size=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "sample_size should be greater than 0."

    for i in range(10):
        spoint = spaceopt.fit_predict(num_spoints=1, num_boost_round=100, sample_size=100)
        assert isinstance(spoint, dict)
        spoint['y'] = 0.95
        spaceopt.append_evaluated_spoint(spoint)


def test_SpaceOpt__sample_random_spoints():
    spaceopt = SpaceOpt(search_space=search_space(), target_name='y', objective='min')

    try:
        spaceopt._sample_random_spoints(sample_size=100.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "sample_size is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt._sample_random_spoints(sample_size=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "sample_size should be greater than 0."

    spoints = spaceopt._sample_random_spoints(sample_size=10)
    assert isinstance(spoints, list)
    assert len(spoints) == 10

    spoints = spaceopt._sample_random_spoints(sample_size=1)
    assert isinstance(spoints, list)
    assert len(spoints) == 1


def test_SpaceOpt__sample_unevaluated_unique_spoints():
    spaceopt = SpaceOpt(search_space=search_space(), target_name='y', objective='min')

    try:
        spaceopt._sample_unevaluated_unique_spoints(sample_size=100.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "sample_size is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt._sample_unevaluated_unique_spoints(sample_size=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "sample_size should be greater than 0."

    try:
        spaceopt._sample_unevaluated_unique_spoints(sample_size=100, max_num_retries=10.0)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "max_num_retries is of type <class 'float'>, but it should be of type <class 'int'>."

    try:
        spaceopt._sample_unevaluated_unique_spoints(sample_size=1, max_num_retries=0)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "max_num_retries should be greater than 0."

    spaceopt = SpaceOpt(search_space={'w': ['W1', 'W2'], 'b': [5]}, target_name='y', objective='min')
    spaceopt.append_evaluated_spoint({'w': 'W1', 'b': 5, 'y': 0.1})
    spoints = spaceopt._sample_unevaluated_unique_spoints(sample_size=100, max_num_retries=100)
    assert len(spoints) == 1
    assert spoints[0]['w'] == 'W2'

    spaceopt.append_evaluated_spoint({'w': 'W2', 'b': 5, 'y': 0.2})
    try:
        spaceopt._sample_unevaluated_unique_spoints(sample_size=100, max_num_retries=100)
    except Exception as e:
        assert isinstance(e, RuntimeError)
        assert str(e) == 'could not sample any new spoints - search_space is fully explored or random sampling was unfortunate.\nsearch_space.size = 2\nnum evaluated spoints = 2\nnum unevaluated spoints = 0'


def test_SpaceOpt__str__():
    spaceopt = SpaceOpt(search_space=search_space(), target_name='y', objective='min')

    assert str(spaceopt) == "SpaceOpt(\n    Space(\n        Variable(\n            name='a',\n            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],\n            vtype=<class 'int'>,\n            is_categorical=False\n        ),\n        Variable(\n            name='b',\n            values=[-5.5, -4.4, -3.3, -2.2, -1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5],\n            vtype=<class 'float'>,\n            is_categorical=False\n        ),\n        Variable(\n            name='c',\n            values=[128, 256, 512, 1024],\n            vtype=<class 'int'>,\n            is_categorical=False\n        ),\n        Variable(\n            name='d',\n            values=['typeX', 'typeY', 'typeZ'],\n            vtype=<class 'str'>,\n            is_categorical=True\n        ),\n        Variable(\n            name='e',\n            values=[True, False],\n            vtype=<class 'bool'>,\n            is_categorical=False\n        ),\n        Variable(\n            name='f',\n            values=[10000],\n            vtype=<class 'int'>,\n            is_categorical=False\n        ),\n        size=5544\n    ),\n    target_name='y',\n    objective=min\n)"
