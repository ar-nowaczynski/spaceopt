import random
from spaceopt.variable import Variable
random.seed(123456)


def test_Variable__init__():
    import numpy as np

    try:
        Variable(name=12, values=[1.0, 4.5, 7.8])
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "Invalid name=12 for a Variable. Provided name is of type <class 'int'>, but it should be of type <class 'str'>."

    try:
        Variable(name=None, values=[1.0, 4.5, 7.8])
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "Invalid name=None for a Variable. Provided name is of type <class 'NoneType'>, but it should be of type <class 'str'>."

    try:
        Variable(name=(1, 2), values=[1.0, 4.5, 7.8])
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "Invalid name=(1, 2) for a Variable. Provided name is of type <class 'tuple'>, but it should be of type <class 'str'>."

    try:
        Variable(name='aa', values=(1.0, 4.5, 7.8))
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "Variable named 'aa' has values=(1.0, 4.5, 7.8) of type <class 'tuple'>, but it should be of type <class 'list'>."

    try:
        Variable(name='aa', values={'a': 1.0, 'b': 4.5, 'c': 7.8})
    except Exception as e:
        assert isinstance(e, TypeError)
        assert str(e) == "Variable named 'aa' has values={'a': 1.0, 'b': 4.5, 'c': 7.8} of type <class 'dict'>, but it should be of type <class 'list'>."

    try:
        Variable(name='bb', values=[])
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "Variable named 'bb' has an empty list of values."

    try:
        Variable(name='cc', values=[np.int32(1), np.int32(5), 7])
    except Exception as e:
        assert isinstance(e, RuntimeError)
        assert str(e) == "Multiple value types for a Variable named 'cc' with values=[1, 5, 7]. Encountered value types:\n1 : <class 'numpy.int32'>\n5 : <class 'numpy.int32'>\n7 : <class 'int'>\nAll values should be of the same type. Allowed value types: (<class 'float'>, <class 'int'>, <class 'str'>, <class 'bool'>)."

    try:
        Variable(name='dd', values=[np.int32(1), np.int32(5), np.int32(7)])
    except Exception as e:
        assert isinstance(e, RuntimeError)
        assert str(e) == "All values=[1, 5, 7] for a Variable named 'dd' are of type <class 'numpy.int32'>, which is not allowed. Please use one of: (<class 'float'>, <class 'int'>, <class 'str'>, <class 'bool'>)."


def test_Variable_is_categorical():
    variable = Variable(name='var', values=['v1', '5', '*'])
    assert variable.is_categorical


def test_Variable__str__():
    assert str(Variable(name='ff', values=[1.0, 4.5, 7.8])) == "Variable(\n    name='ff',\n    values=[1.0, 4.5, 7.8],\n    vtype=<class 'float'>,\n    is_categorical=False\n)"

    assert str(Variable(name='bb', values=[True, False])) == "Variable(\n    name='bb',\n    values=[True, False],\n    vtype=<class 'bool'>,\n    is_categorical=False\n)"

    assert str(Variable(name='ii', values=[32, 64, 128, 256, 512])) == "Variable(\n    name='ii',\n    values=[32, 64, 128, 256, 512],\n    vtype=<class 'int'>,\n    is_categorical=False\n)"

    assert str(Variable(name='cc', values=['x', 'y', 'z'])) == "Variable(\n    name='cc',\n    values=['x', 'y', 'z'],\n    vtype=<class 'str'>,\n    is_categorical=True\n)"
