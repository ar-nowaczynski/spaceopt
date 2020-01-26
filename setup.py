from setuptools import setup

setup(name='spaceopt',
      version='0.1.0',
      description='Search space optimization via predictive modelling',
      author='Arkadiusz Nowaczynski',
      author_email='ar.nowaczynski@gmail.com',
      url='https://github.com/ar-nowaczynski/spaceopt',
      packages=['spaceopt'],
      install_requires=[
          'lightgbm>=2.3.0',
          'pandas>=0.25.3',
      ])
