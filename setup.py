from setuptools import setup, find_packages
from os import path

__version__ = '0.0.1'

path = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(path, 'requirements.txt'), encoding='utf-8') as file:
    requirements = file.read().split('\n')

install_requires = [x.strip() for x in requirements if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in requirements if x.startswith('git+')]

setup(
    name='deepagent',
    version=__version__,
    author='Kashu Yamazaki',
    author_email='kyamazak@uark.edu',
    description='Python implementations of Deep Learning models and algorithms with a minimum use of external library.',
    url='https://github.com/kashu98/Deep-Agent',
    download_url='https://github.com/kashu98/Deep-Agent/tarball/master',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=['numpy>=1.13'],
    dependency_links=dependency_links,
)
