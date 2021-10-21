import setuptools

NAME = 'pmap'
VERSION = '0.0.1'
if __name__ == '__main__':
    setuptools.setup(
        name=NAME,
        version=VERSION,
        packages=setuptools.find_namespace_packages('pmap'),
    )
