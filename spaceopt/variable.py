import random
from collections import Counter


class Variable:

    _ALLOWED_VTYPES = (float, int, str, bool)

    def __init__(self, name, values):
        self._verify_name(name)
        self.name = name
        self._verify_values(values)
        self.values = values
        self.vtype = self._get_vtype_from_values()

    @property
    def is_categorical(self):
        return self.vtype is str

    def sample(self):
        return random.choice(self.values)

    def encode(self, df):
        if self.is_categorical:
            encoding = dict(zip(self.values, range(len(self.values))))
            df[self.name] = df[self.name].map(encoding)
        return df

    def decode(self, df):
        if self.is_categorical:
            decoding = dict(zip(range(len(self.values)), self.values))
            df[self.name] = df[self.name].map(decoding)
        return df

    def _verify_name(self, name):
        if not isinstance(name, str):
            raise TypeError(f'Invalid name={name} for a {self.__class__.__name__}. '
                            f'Provided name is of type {type(name)}, '
                            f'but it should be of type {str}.')

    def _verify_values(self, values):
        if not isinstance(values, list):
            raise TypeError(f'{self.__class__.__name__} named {repr(self.name)} '
                            f'has values={values} '
                            f'of type {type(values)}, '
                            f'but it should be of type {list}.')
        if len(values) == 0:
            raise ValueError(f'{self.__class__.__name__} named {repr(self.name)} '
                             f'has an empty list of values.')

    def _get_vtype_from_values(self):
        vtypes = [type(value) for value in self.values]
        cnt = Counter(vtypes)
        if len(cnt) != 1:
            value_types = '\n'.join([f'{v} : {type(v)}' for v in self.values])
            raise RuntimeError(f'Multiple value types for a {self.__class__.__name__} '
                               f'named {repr(self.name)} '
                               f'with values={self.values}. '
                               f'Encountered value types:\n{value_types}\n'
                               f'All values should be of the same type. '
                               f'Allowed value types: {self._ALLOWED_VTYPES}.')
        vtype = cnt.most_common()[0][0]
        if vtype not in self._ALLOWED_VTYPES:
            raise RuntimeError(f'All values={self.values} for a {self.__class__.__name__} '
                               f'named {repr(self.name)} '
                               f'are of type {vtype}, which is not allowed. '
                               f'Please use one of: {self._ALLOWED_VTYPES}.')
        return vtype

    def __str__(self):
        indent = ' ' * 4
        innerstr = [
            'name={}'.format(repr(self.name)),
            'values={}'.format(self.values),
            'vtype={}'.format(self.vtype),
            'is_categorical={}'.format(self.is_categorical),
        ]
        innerstr = indent + (',\n' + indent).join(innerstr)
        outstr = '{cls}(\n{innerstr}\n)'.format(
            cls=self.__class__.__name__,
            innerstr=innerstr,
        )
        return outstr
