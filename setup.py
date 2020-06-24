from setuptools import setup, find_packages, Extension
import os

# Set compiler to g++
os.environ["CXX"] = "g++"

# Path to where the header of the GNU Scientific Library is installed.
path_to_gsl_header = '/usr/include'

setup(name='risk_assess',
      version='0.1',
      description='A Risk Assessment Package',
      author='Allen Wang',
      author_email='allenw@mit.edu',
      license='MIT',
      packages=find_packages(),
      ext_modules=[Extension('imhof', ['risk_assess/cpp/imhof.cpp'], include_dirs = [path_to_gsl_header],
                              extra_compile_args = ["-O3"],
                              extra_link_args = ["-O3", "-lgsl", "-lgslcblas", "-lm"]),])