# Go to the repo root 
cd $(git rev-parse --show-toplevel)

# Install the GNU Scientific Library
sudo apt-get update
sudo apt-get install gsl-bin libgsl0-dev
sudo apt-get install g++

# Create the virtual environment
python3 -m pip install --user virtualenv
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt

# Install the argoverse-api, which is included as
# a submodule
git submodule update --init --recursive
cd argoverse-api
git checkout master
python -m pip install -e .
cd ..


# Make a temporary directory for output.
mkdir /tmp/argoverse
