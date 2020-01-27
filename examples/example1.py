from spaceopt import SpaceOpt


def feval(spoint):
    ''' Synthetic evaluation function.

    Global minimum is at:
    {
        'a': 5.0,
        'b': -5.5,
        'c': 1024,
        'd': 'typeY',
        'e': 'False',
        'f': 10000,
    },
    with value:
        'y': -6.29.

    '''
    a = spoint['a']
    b = spoint['b']
    c = spoint['c']
    d = spoint['d']
    e = spoint['e']
    f = spoint['f']
    xa = abs(a - 5) / 10
    xb = b
    xc = {
        128: 0.7,
        256: 0.72,
        512: 0.68,
        1024: 0.78,
    }[c]
    xd = {
        'typeX': -1,
        'typeY': -2,
        'typeZ': 0,
    }[d]
    xe = 0.1234 if e else 0.0
    assert f == 10000
    y = (xa + xb) * xc + xd + xe
    return y


def search_space():
    return {
        'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'b': [-5.5, -4.4, -3.3, -2.2, -1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5],
        'c': [128, 256, 512, 1024],
        'd': ['typeX', 'typeY', 'typeZ'],
        'e': [True, False],
        'f': [10000],
    }


def main():

    spaceopt = SpaceOpt(search_space=search_space(), target_name='y', objective='min')
    print(spaceopt)

    best_spoint = None
    best_y = 123456789.0

    for iteration in range(1, spaceopt.space.size + 1):

        if iteration <= 10:
            spoint = spaceopt.get_random(num_spoints=1, sample_size=1000)
            spoint_type = 'random'
        else:
            spoint = spaceopt.fit_predict(num_spoints=1, sample_size=1000)
            spoint_type = 'fit_predict'

        spoint['y'] = feval(spoint)
        spaceopt.append_evaluated_spoint(spoint)

        if spoint['y'] < best_y:
            best_y = spoint['y']
            best_spoint = spoint
            print(f'{iteration}, y={round(best_y, 6)}, {spoint_type}, {best_spoint}')

        if best_y < -6.289:
            print('global minimum has been found')
            print('{}/{} = {}'.format(iteration,
                                      spaceopt.space.size,
                                      iteration / spaceopt.space.size))
            return


if __name__ == '__main__':
    main()
