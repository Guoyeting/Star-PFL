This repository includes source code for the paper Y. Guo, F. Liu, T. Zhou, Z. Cai and N. Xiao. "Efficient personalized federated learning on selective model training".

#### Requirements

The code runs on Python 3.8 To install the dependencies, run
```
pip3 install -r requirements.txt
```

#### Dataset supported
Then, download the datasets manually and put them into the `data` folder.

- OrganMNIST Sagital: downloaded from (https://dgresearchredmond.blob.core.windows.net/amulet/data/cycfed/medmnist.tar.gz)
- OrganMNIST Axial:downloaded from (https://dgresearchredmond.blob.core.windows.net/amulet/data/cycfed/medmnistA.tar.gz)

#### Training
Run the command like `python main.py --dataset medmnistA --iters 100 --wk_iters 5 --non_iid_alpha 0.1`

#### Code Structure
- Directory `alg` includes the core function of Star-PFL
- Directory `data` stores the datasets we used
- Directory `datautil` is used to simulate the heteregenours data settings
- Directory `network` describes the model structure
- Directory `results` records the results
- Directory `split` stores the data distribution information

