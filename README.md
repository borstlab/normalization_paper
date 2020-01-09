# Dynamic Signal Compression for Robust Motion Vision in Flies

Drews, M., Leonhardt, A., Pirogova, N., Richter, F., Schuetzenberger, A., Serbe, E., Braun, L., and Borst, A. Dynamic Signal Compression for Robust Motion Vision in Flies. Current Biology (2020) https://doi.org/10.1016/j.cub.2019.10.035

## Abstract

Sensory systems need to reliably extract information from highly variable natural signals. Flies, for instance, use optic flow to guide their course and are remarkably adept at estimating image velocity regardless of image statistics. Current circuit models, however, cannot account for this robustness. Here, we demonstrate that the Drosophila visual system reduces input variability by rapidly adjusting its sensitivity to local contrast conditions. We exhaustively map functional properties of neurons in the motion detection circuit and find that local responses are compressed by surround contrast. The compressive signal is fast, integrates spatially, and derives from neural feedback. Training convolutional neural networks on estimating the velocity of natural stimuli shows that this dynamic signal compression can close the performance gap between model and organism. Overall, our work represents a comprehensive mechanistic account of how neural systems attain the robustness to carry out survival-critical tasks in challenging real-world environments.

## Code & Data

This repository provides some code and data from the paper:

* Calcium imaging: Raw data from calcium imaging experiments is available in pickled Pandas DataFrames. Instructions for accessing them is found in the corresponding README file.

* Convolutional network experiments: Code for defining and training the convolutional neural networks used in Fig. 6 are found in the subfolder `cnn`.

## Contact

If you have questions regarding the code, data, or paper, don't hesitate to contact either Aljoscha Leonhardt (leonhardt@neuro.mpg.de) or Michael Drews (drews@neuro.mpg.de).
