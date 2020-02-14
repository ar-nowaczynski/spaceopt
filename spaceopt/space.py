from spaceopt.variable import Variable


class Space:

    def __init__(self, search_space):
        self._verify_search_space(search_space)
        self.variables = []
        for name, values in search_space.items():
            variable = Variable(name=name, values=values)
            self.variables.append(variable)

    @property
    def size(self):
        size = 1
        for variable in self.variables:
            size *= len(variable.values)
        return size

    @property
    def variable_names(self):
        return [variable.name for variable in self.variables]

    @property
    def categorical_names(self):
        return [variable.name for variable in self.variables if variable.is_categorical]

    def sample(self):
        return {variable.name: variable.sample() for variable in self.variables}

    def encode_variables(self, df):
        for variable in self.variables:
            df = variable.encode(df)
        return df

    def decode_variables(self, df):
        for variable in self.variables:
            df = variable.decode(df)
        return df

    def verify_spoint(self, spoint):
        for variable in self.variables:
            if variable.name not in spoint:
                raise ValueError(f'spoint={spoint} should have variable '
                                 f'named {repr(variable.name)}.')
            if not isinstance(spoint[variable.name], variable.vtype):
                raise TypeError(f'spoint has variable '
                                f'named {repr(variable.name)} '
                                f'with value {spoint[variable.name]} '
                                f'of type {type(spoint[variable.name])}, '
                                f'but it should be of type {variable.vtype}.')
            if spoint[variable.name] not in variable.values:
                raise ValueError(f'spoint has variable '
                                 f'named {repr(variable.name)} '
                                 f'with value={spoint[variable.name]}, which is '
                                 f'outside of the defined list of values={variable.values}.')

    def _verify_search_space(self, search_space):
        if not isinstance(search_space, dict):
            raise TypeError(f'search_space is of type {type(search_space)}, '
                            f'but it should be of type {dict}.')
        if len(search_space) == 0:
            raise ValueError('search_space is empty.')

    def __str__(self):
        indent = ' ' * 4
        innerstr = []
        innerstr += [str(variable).replace('\n', '\n' + indent)
                     for variable in self.variables]
        innerstr += [f'size={self.size}']
        innerstr = indent + (',\n' + indent).join(innerstr)
        outstr = '{cls}(\n{innerstr}\n)'.format(
            cls=self.__class__.__name__,
            innerstr=innerstr,
        )
        return outstr
