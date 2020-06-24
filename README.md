# RiskAssess
Code for:
Wang, Allen, et al. "Fast Risk Assessment for Autonomous Vehicles Using Learned Models of Agent Futures." arXiv preprint arXiv:2005.13458 (2020).

We are currently in the process of organizing and cleaning up our code for presentation to the public. Expect this to be ready by the time the paper is published in Robotics: Science and Systems!

## Data
#### Filter and subsampling data
Let $REPO_ROOT be the path to the root of the repo.
```bash
python scripts/filter_argoverse_data.py -i $REPO_ROOT/dataset/argoverse_raw/forecasting_sample/data/ -o $REPO_ROOT/dataset/argoverse_filtered -l 500
```
*l* specifies the number of samples to keep.

#### Train a model
Go to project directory, and run:
```bash
tensorboard --logdir=logs/
python prediction/train.py --data_dir dataset/argoverse_filtered/ --log_dir=logs/
```
See tensorboard logs using the link given after running the first command.

### Contact
Cyrus Huang, xhuang at csail dot mit dot edu