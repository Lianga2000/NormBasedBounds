# Norm-based Generalization Bounds for Compositionally Sparse Neural Networks

## Requirements
- Python 3.10
- Pytorch 1.11
- Numpy
- Tqdm

## Running Experiments

**To submit the code as a job to slurm:**
    ```
    sh train.sh
    ```

Other Files

* conf/global_settings.py: A file that specifies the configuration parameters and hyperparameters.
* utils.py: Contains functions responsible for saving data, loading datasets, measuring our generalization bounds, and alternative bounds from the literature, etc.
* models: Contains implementations of networks used in training.

<br />
<hr> 
<h3> Citation </h3>

```bib
@article{galanti2024norm,
  title={Norm-based generalization bounds for sparse neural networks},
  author={Galanti, Tomer and Xu, Mengjia and Galanti, Liane and Poggio, Tomaso},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
