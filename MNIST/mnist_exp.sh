# make sure notMNIST dataset is downloaded and
# saved under data/data_notMNIST_small.npz.
python prepare_data.py
python train.py
mkdir figs/
python eval_in_domain.py
python eval_ood.py
python eval_distribution_shift.py
