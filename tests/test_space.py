import random
from spaceopt.space import Space
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


def test_Space__init__():
    try:
        Space(search_space={'a', 'l', 'z'})
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "search_space is of type <class 'set'>, but it should be of type <class 'dict'>."

    try:
        Space(search_space={})
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "search_space is empty."


def test_Space_size():
    space = Space(search_space=search_space())
    assert space.size == 5544


def test_Space_variable_names():
    space = Space(search_space=search_space())
    assert set(space.variable_names) == set(search_space().keys())


def test_Space_categorical_names():
    space = Space(search_space=search_space())
    assert set(space.categorical_names) == {'d'}


def test_Space_verify_spoint():
    space = Space(search_space=search_space())

    spoint = {'a': 16, 'b': 3.3, 'd': 'typeX', 'e': False}
    try:
        space.verify_spoint(spoint)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "spoint={'a': 16, 'b': 3.3, 'd': 'typeX', 'e': False} should have variable named 'c'."

    spoint = {'a': 16, 'b': 3.3, 'c': 512.0, 'd': 'typeX', 'e': False}
    try:
        space.verify_spoint(spoint)
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "spoint has variable named 'c' with value 512.0 of type <class 'float'>, but it should be of type <class 'int'>."

    spoint = {'a': 16, 'b': 3.3, 'c': 700, 'd': 'typeX', 'e': False}
    try:
        space.verify_spoint(spoint)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "spoint has variable named 'c' with value=700, which is outside of the defined list of values=[128, 256, 512, 1024]."


def test_Space__str__():
    assert str(Space(search_space=search_space())) == "Space(\n    Variable(\n        name='a',\n        values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],\n        vtype=<class 'int'>,\n        is_categorical=False\n    ),\n    Variable(\n        name='b',\n        values=[-5.5, -4.4, -3.3, -2.2, -1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5],\n        vtype=<class 'float'>,\n        is_categorical=False\n    ),\n    Variable(\n        name='c',\n        values=[128, 256, 512, 1024],\n        vtype=<class 'int'>,\n        is_categorical=False\n    ),\n    Variable(\n        name='d',\n        values=['typeX', 'typeY', 'typeZ'],\n        vtype=<class 'str'>,\n        is_categorical=True\n    ),\n    Variable(\n        name='e',\n        values=[True, False],\n        vtype=<class 'bool'>,\n        is_categorical=False\n    ),\n    Variable(\n        name='f',\n        values=[10000],\n        vtype=<class 'int'>,\n        is_categorical=False\n    ),\n    size=5544\n)"
