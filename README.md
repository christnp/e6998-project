# On Finding the Best Strategy for Limited Dataset Deep Learning

## Introduction

TBC

The report that accompanies this work can be found in this repo: [ADD PAPER LINK HERE]()


### Objective
The goal of the experiments covered in this repo is to simply provide a survey of some different deep learning approaches that attempt or claim to address the issue of training models on limited (good/labelled) data availability.

### Environment
There would be no code if it weren't for the amazing infrastructure and frameworks. This section provides a glance at the tools we leveraged to complete the objective defined above. We recommended the following dependencies:
* `Python 3.7.8`
* `PyTorch 1.4.0`

<table>
    <tr>
        <th><img src="resources/pytorch.svg" height="70" style="padding: 10px 10px 10px 10px;"></th>
        <th><img src="resources/python-logo-generic.svg" height="75" style="padding: 10px 10px 10px 10px;"></th>
    </tr>
</table>


We ran all of our experiments on the Google Cloud Platform infrastructure, leveraging the Google AI Platform as the Jupyter notebook server.

<table>
    <tr>
        <th><img src="resources/google-ai-platform.svg" height="65" style="padding: 10px 10px 10px 10px;"></th>
        <th><img src="resources/google-cloud-platform.svg" height="65"style="padding: 10px 10px 10px 40px;"></th>
    </tr>
</table>

Finally, so that we remained consistent for all experiment, we used the `n1-standard-8` preconfigured hardware (8 vCPUs, 30 GB RAM) as our host machine for running Jupyter notebook. When training the final model weights, we used the `NVIDIA Tesla V100` GPU to speed up training time.

### Repo structure
This repository is organized in the following way:

    /src
      |-- augmentation
      |     |-- TBD ...      
      |     |-- models
      |     |-- results
      |-- transfer-learning
      |     |-- <same>      
      |-- one-shot
      |     |-- <same>      
      |-- zero-shot
      |     |-- <same>      
    
    /data 
      |-- MNIST <-- not stored in git
      |     |-- train
      |     |-- validate
      |     |-- test

### Running the code
The code execution for all tasks is similar. 
```terminal
python one-shot/oneshot.py -m ... -n ... etc.
```
Alternatively, the code be executed in a Jupyter notebook (simply import the notebook from this repo!).

```python
import torch
import torch.vision

def do(this)
    return do_that


```
   
## Experiments
This section provides more details for each of the DL tasks covered in this repository.   
   
### Data Augmentation [*](src/augmentation)
Data agumentation stuff
   
### Transfer Learning [*](src/transfer-learning)
Transfer learning stuff
   
### One-Shot Learning [*](src/one-shot)
One-shot learning stuff
   
### Zero-Shot Learning [*](src/zero-shot)
Zero-shot learning stuff





## References:
[1] Li, Kai, et al. “Rethinking Zero-Shot Learning: A Conditional Visual Classification Perspective.” ArXiv:1909.05995 [Cs], Nov. 2019. arXiv.org, http://arxiv.org/abs/1909.05995. 