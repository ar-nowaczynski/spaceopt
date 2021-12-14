from typing import Dict
from spaceopt import SpaceOpt


def evaluation_function(spoint: dict) -> float:
    ''' Synthetic evaluation function.

    Global minimum is at:

        {
            'a': 5.0,
            'b': -5.5,
            'c': 1024,
            'd': 'val_B',
            'e': 'False',
            'f': 10000,
        }

    with value:

        'y': -6.29

    '''
    xa = abs(spoint['a'] - 5) / 10
    xb = spoint['b']
    xc = {128: 0.7, 256: 0.72, 512: 0.68, 1024: 0.78}[spoint['c']]
    xd = {'val_A': -1, 'val_B': -2, 'val_C': 0}[spoint['d']]
    xe = 0.1234 if spoint['e'] else 0.0
    assert spoint['f'] == 10000
    y = (xa + xb) * xc + xd + xe
    return y


def search_space() -> Dict[str, list]:
    return {
        'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'b': [-5.5, -4.4, -3.3, -2.2, -1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5],
        'c': [128, 256, 512, 1024],
        'd': ['val_A', 'val_B', 'val_C'],
        'e': [True, False],
        'f': [10000],
    }


def main() -> None:
    spaceopt = SpaceOpt(search_space=search_space(),
                        target_name='y',
                        objective='minimize')
    print(spaceopt)

    best_spoint = None
    best_y = float('inf')

    for iteration in range(1, spaceopt.space.size + 1):
        if iteration <= 10:
            # exploration
            spoint = spaceopt.get_random(num_spoints=1, sample_size=1000)
            spoint_type = 'random'
        else:
            # exploitation
            spoint = spaceopt.fit_predict(num_spoints=1, sample_size=1000)
            spoint_type = 'fit_predict'

        spoint['y'] = evaluation_function(spoint)
        spaceopt.append_evaluated_spoint(spoint)

        if spoint['y'] < best_y:
            best_y = spoint['y']
            best_spoint = spoint
            print(f'{iteration}, y={round(best_y, 6)}, {spoint_type}, {best_spoint}')

        if best_y < -6.289:
            print('global minimum has been found\n'
                  f'{iteration}/{spaceopt.space.size} = '
                  f'{iteration / spaceopt.space.size}')
            return


if __name__ == '__main__':
    main()
