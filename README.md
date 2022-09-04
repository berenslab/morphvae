# MorphVAE: Generating Neural Morphologies from 3D-Walks using Variational Autoencoder with Spherical Latent Space

This repository stores the code related to the ICML2021 [paper](https://proceedings.mlr.press/v139/laturnus21a.html).

![](https://github.com/berenslab/morphvae/blob/master/Fig1.png "Model schematic")

## Running the notebooks

### Overview of the notebooks
**Data processing**

- Download data: 
    Prior to running the notebooks you will need to download the associated morphologies as well as processed data. You can find the data [here](https://doi.org/10.5281/zenodo.4920391). Please download and unpack into the same repository. 
- [Create Toy data](https://github.com/berenslab/morphvae/blob/master/Create%20Toy%20data.ipynb): Generates toy data and their random walk representation.
- [Create data iterators](https://github.com/berenslab/morphvae/blob/master/Create%20data%20iterators.ipynb): Creates the data iterators for model fitting. Note, the random walk representation has been pre-generated and uploaded in the data repository. If you want to generate them yourself, you can find the code in the [./utils/rw_utils.py](https://github.com/berenslab/morphvae/blob/master/utils/rw_utils.py#L39)
- [Create image stacks](https://github.com/berenslab/morphvae/blob/master/Create%20image%20stacks.ipynb): Creates an image stack for each neuron to be fed into the TREES Toolbox
- [Density maps on toy data and real data](https://github.com/berenslab/morphvae/blob/master/Density%20maps%20on%20toy%20data%20and%20real%20data.ipynb): Analysis pipeline of density map projections for each data set. Creation (1D, 2D and 3D projections), 2D t-SNE visualisation and 5-NN classification.

**Model fitting**

Training:

  - [Train vae on toy data](https://github.com/berenslab/morphvae/blob/master/Train%20vae%20on%20toy%20data.ipynb)
  - [Train vae from scratch on Farrow data](https://github.com/berenslab/morphvae/blob/master/Train%20vae%20from%20scratch%20on%20Farrow%20data-using%20scaling.ipynb)
  - [Train with shuffled labels](https://github.com/berenslab/morphvae/blob/master/Train%20with%20shuffled%20labels.ipynb)
  - [Pre-train vae with scaling data augmentation](https://github.com/berenslab/morphvae/blob/master/Pre-train%20vae%20with%20scaling%20data%20augmentation.ipynb) (basis for the finetuning of the real world data)

Finetune pre-tained models (on toy data) on real data: 

  - [Finetune vae on EXC data -mtype labels](https://github.com/berenslab/morphvae/blob/master/Finetune%20vae%20on%20EXC%20data-mtype%20labels.ipynb)
  - Finetune van on Farrow data 
    - [pre-tained on scaled Farrow data](https://github.com/berenslab/morphvae/blob/master/Finetune%20vae%20on%20Farrow%20data%20-%20pretraining%20on%20scaled%20data.ipynb) (from scratch)
    - [pre-trained on scaled toy data](https://github.com/berenslab/morphvae/blob/master/Finetune%20vae%20on%20Farrow%20data%20-%20pretraining%20on%20toy%20data.ipynb)
  - [Finetune van on INH data - rna labels](https://github.com/berenslab/morphvae/blob/master/Finetune%20vae%20on%20INH%20data%20-%20rna%20label%20-%20axon.ipynb)

**Analysis**

  - [Evaluate parameter search](https://github.com/berenslab/morphvae/blob/master/Evaluate%20parameter%20search.ipynb)
  - [Evaluate runtime](https://github.com/berenslab/morphvae/blob/master/Evaluate%20runtime.ipynb)
  - [Get morphometric statistic](https://github.com/berenslab/morphvae/blob/master/Get%20morphometric%20statistics.ipynb)
  - [Results Ablation study](https://github.com/berenslab/morphvae/blob/master/Results%20ablation%20study%20.ipynb)

Classification:

  - [Classification analysis - how many labels are needed](https://github.com/berenslab/morphvae/blob/master/Classification%20analysis%20-%20how%20many%20labels%20are%20needed.ipynb): Analysis on toy data
  - [K-nearest neighbour classification on neuron rep](https://github.com/berenslab/morphvae/blob/master/K-nearest%20neighbor%20classification%20on%20neuron%20rep.ipynb): 5-NN classification and confusion matrices for each data set

Sample new neurons:

  - [Cluster walks and sample neurons](https://github.com/berenslab/morphvae/blob/master/Cluster%20walks%20and%20sample%20neurons.ipynb): Sample a new walk representation from existing neutron, cluster those walks and reconstruct a new neuron on all data sets
  - [Sample neurons](https://github.com/berenslab/morphvae/blob/master/Sample%20neurons.ipynb): Sample new neurons for each data set
	
**Plotting**

  - [Plot M1 data](https://github.com/berenslab/morphvae/blob/master/Plot%20M1%20data.ipynb)
  - [Plot Farrow data](https://github.com/berenslab/morphvae/blob/master/Plot%20Farrow%20data.ipynb)
  - [Plot latent space of walks](https://github.com/berenslab/morphvae/blob/master/Plot%20latent%20space%20of%20walks.ipynb): 2D embedding of walk representation learned under each data set
  - [Plot t-SNE of neural representation](https://github.com/berenslab/morphvae/blob/master/Plot%20t-SNE%20of%20neural%20representations.ipynb): 2D embeddings of the neural representation learned by each model on each data set. 


**Misc**

  - [Get system resources](https://github.com/berenslab/morphvae/blob/master/Get%20system%20resources.ipynb)
  - [Sample walks](https://github.com/berenslab/morphvae/blob/master/Sample%20walks.ipynb)

## Citing our work 

If you use or refer to this work please use the following citation:
```

@InProceedings{pmlr-v139-laturnus21a,
  title = 	 {MorphVAE: Generating Neural Morphologies from 3D-Walks using a Variational Autoencoder with Spherical Latent Space},
  author =       {Laturnus, Sophie C. and Berens, Philipp},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {6021--6031},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/laturnus21a/laturnus21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/laturnus21a.html},
}

```
