# Install the GNU Scientific Library
sudo apt-get update
sudo apt-get install gsl-bin libgsl0-dev
sudo apt-get install g++

# Activate the virtual environment and install this package.
source venv/bin/activate
pip install -e .

# Make a temporary directory for output.
mkdir /tmp/argoverse