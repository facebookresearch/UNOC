## UNOC: Understanding Occlusion for Embodied Presence in Virtual Reality

IEEE Trans Vis Comput Graph. 2022 Dec.

*Mathias Parger, Chengcheng Tang, Yuanlu Xu, Christopher D Twigg, Lingling Tao, Yijing Li, Robert Wang, Markus Steinberger*

<p float="left">
<img src="./images/1.png" width="10%"/>
<img src="./images/2.png" width="10%"/>
<img src="./images/3.png" width="10%"/>
<img src="./images/4.png" width="10%"/>
<img src="./images/5.png" width="10%"/>
<img src="./images/6.png" width="10%"/>
<img src="./images/7.png" width="10%"/>
<img src="./images/8.png" width="10%"/>
<img src="./images/9.png" width="10%"/>
<p>

### Abstract

Tracking body and hand motions in 3D space is essential for social and self-presence in augmented and virtual environments. Unlike the popular 3D pose estimation setting, the problem is often formulated as egocentric tracking based on embodied perception (e.g., egocentric cameras, handheld sensors). In this article, we propose a new data-driven framework for egocentric body tracking, targeting challenges of omnipresent occlusions in optimization-based methods (e.g., inverse kinematics solvers). We first collect a large-scale motion capture dataset with both body and finger motions using optical markers and inertial sensors. This dataset focuses on social scenarios and captures ground truth poses under self-occlusions and body-hand interactions. We then simulate the occlusion patterns in head-mounted camera views on the captured ground truth using a ray casting algorithm and learn a deep neural network to infer the occluded body parts. Our experiments show that our method is able to generate high-fidelity embodied poses by applying the proposed method to the task of real-time egocentric body tracking, finger motion synthesis, and 3-point inverse kinematics.

### Usage

Please download the [UNOC dataset](https://cloud.tugraz.at/index.php/s/ykqxA7HxYMnwbXr) and extract the BVH files.

Install Python and Pytorch. Please follow the Pytorch installation instructions from their [website](https://pytorch.org/get-started/locally/).

Install dependencies by running `pip install -r requirements.txt`

Before running any scripts, please set the environment variables

- _PATH_UNOC_ to the root directory of the UNOC dataset (containing the 13 participants).
- _PATH_DATA_ to a directory that can be used for temporary data, such as preprocessed feature sets and model weights.

#### Training

Once the environment variables are set, you can run the training script _train.py_. At first start, this will process the bvh files and save the pose
information as numpy files in _UNOC_PATH_. After the conversion is done, the feature sets for input and output are created and saved to _PATH_DATA_ to speed up
training. The training procedure will start once all feature sets are converted and should finish after a few minutes.

_train.py_ can be run in two modes:

- Predicting the fully body pose from incomplete tracking information `python train.py body`
- Predicting finger pose from body pose `python train.py finger`

#### Evaluation

Like _train.py_, _eval.py_ can be run in two modes:

- Predicting the fully body pose from incomplete tracking information `python eval.py body`
- Predicting finger pose from body pose `python eval.py finger`

Both modes support _plot_ as an additional argument that will animate the predicted animations using an interactive 3D view.

### Dataset

Try out our [interactive web player](https://dabeschte.github.io/UNOC-Demo/) to preview some examples of UNOC motion.

The dataset is available in different file formats and at different stages of processing:

- [Solved and calibrated body and hand pose (BVH)](https://cloud.tugraz.at/index.php/s/ykqxA7HxYMnwbXr): This data is used for training and evaluating the
  neural network.
- [Solved body without hand motion (FBX)](https://cloud.tugraz.at/index.php/s/NjdGBey6zaiayAb)
- [Solved hand motion (BVH)](https://cloud.tugraz.at/index.php/s/x4aorYcgtXoCLSL): Motion of hands attached to a static avatar. Beware that the wrist rotation is not accurate and only the finger joint
  motion is used in the merged animation.
- [Cleaned body markers (C3D)](https://cloud.tugraz.at/index.php/s/q7KQ5gkkeP5QGoS): We used
  the [Optitrack Biomech 57](https://v22.wiki.optitrack.com/index.php?title=Biomech_(57)) markerset.
  
## License

`UNOC` is released under the MIT license. See [LICENSE](LICENSE.txt) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
