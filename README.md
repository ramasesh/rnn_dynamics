# Dynamics of text-classification RNNs

See results in [arXiv:2010.15114](https://arxiv.org/abs/2010.15114).  This repo contains code (written in JAX) for training networks on various text-classification tasks and analyzing the learned dynamical systems.  

**Note**: This repo is in the process of being converted to open-source, but still has artifacts left over from the earlier, more messy state, e.g. the name of the cloud storage bucket.  These should be fixed soon.

## Running locally

Install dependencies using `make` or `pip`:
```
make install
``` 
will create a VirtualEnv in the project root with all the required packages, or 
```
pip install -r pip_requirements.txt
```
can be used to install the dependencies in an existing VirtualEnv.

## Training

To train a model, run a command like the following:
```
python -m src.train --cell_type GRU --emb_size 128 --num_units 256 --dataset imdb 
```

### Training on subsets of full datasets

Ordered datasets in this study, **Yelp** and **Amazon** reviews, have examples divided into five classes (the number of stars the user left with their review).  This dataset can be coarse-grained into three classes (keeping one-star, three-star, and five-star reviews), or two classes (grouping one-star and two-star reviews together, and four-star and five-star reviews together).  Categorical datasets in this study, **AG News** and **DBPedia** have 4 and 14 classes, respectively.  These can also be reduced down to 3 classes.  In both cases this is done using the `--num_classes` flag.
