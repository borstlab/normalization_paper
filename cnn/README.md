# Convolutional model of fly motion detection

Aljoscha Leonhardt (leonhardt@neuro.mpg.de, Borst Lab, MPI of Neurobiology)

This code implements a simple convolutional model of the fly visual system in PyTorch. Model definitions are found in `model.py`; training happens in `run_training.py`. Tools & helpers for stimulus generation are available in `deepfly`. The full synthetic data set is available upon request.

The model requires

* PyTorch 0.4.1
* NumPy
* tqdm
* TensorBoard

Training can be initiated as follows

`run_training.py --n_epochs 800 --device cuda:0 --batch_size 128 --learning_rate 0.025 --weight_decay 0.0 --noise_factor 0.0 --logdir logdir --normalization_mode none`

and takes between 8-12h on a single NVIDIA Titan X (depending on parameters).
