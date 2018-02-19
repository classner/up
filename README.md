# Unite the People code repository

Requirements:

* OpenCV (on Ubuntu, e.g., install libopencv-dev and python-opencv).
* SMPL (download at http://smpl.is.tue.mpg.de/downloads) and unzip to a 
  place of your choice.
* OpenDR (just run `pip install opendr`, unfortunately can't be done
  automatically with the setuptools requirements.
* If you want to train a segmentation model, Deeplab V2
  (https://bitbucket.org/aquariusjay/deeplab-public-ver2) with a minimal patch
  applied that can be found in the subdirectory `patches`, to enable on the fly
  mirroring of the segmented images. Since I didn't use the MATLAB interface and
  did not care about fixing related errors, I just deleted
  `src/caffe/layers/mat_{read,write}_layer.cpp` as well as
  `src/caffe/util/matio_io.cpp` and built with `-DWITH_matlab=Off`.
* If you want to train a pose model, the Deepercut caffe
  (https://github.com/eldar/deepcut-cnn).
* If you want to get deepercut-cnn predictions, download the deepercut
  .caffemodel file and place it in models/pose/deepercut.caffemodel.
* Edit the file `config.py` to set up the paths.

The rest of the requirements is then automatically installed when running:

```
python setup.py develop
```

## Folder structure

For each of the tasks we described, there is one subfolder with the related
executables. All files that are being used for training or testing models are
executable and provide a full synposis when run with the `--help` option. In the
respective `tools` subfolder for each task, there is a `create_dataset.py`
script to summarize the data in the proper formats. This must be usually run
before the training script. The `models` folder contains pretrained models and
infos, `patches` a patch for deeplab caffe, `tests` some Python tests and
`up_tools` some Python tools that are shared between modalities.

There is a Docker image available that has been created by TheWebMonks here (not
affiliated with the authors): https://github.com/TheWebMonks/demo-2d3d .

### Bodyfit

The adjusted SMPLify code to fit bodies to 91 keypoints is located in the folder
`3dfit`. It can be used for 14 or 91 keypoints. Use the script `3dfit/render.py`
to render a fitted body.

### Direct 3D fitting using regression forests

The relevant files are in the folder `direct3d`. Run
`run_partforest_training.sh` to train all regressors. After that, you can use
`bodyfit.py` to get predictions from estimated keypoints of the 91 keypoint pose
predictor.

### 91 keypoint pose prediction

The `pose` folder containes infrastructure for 91 keypoint pose prediction. Use
the script `pose/tools/create_dataset.py` with a dataset name of your choice and
a target person size of 500 pixels to create the pose data from UP-3D,
alternatively download it from our [website](http://up.is.tuebingen.mpg.de).

Configure a model by creating the model configuration folder
`pose/training/config/modelname` by cloning the `pose` model. Then you can run
`run.sh {train,test,evaluate,trfull,tefull,evfull} modelname` to run training,
testing or evaluation on either the reduced training set with the held-out
validation set as test data or the full training set and real test data. We
initialized our training from the original Resnet models
(https://github.com/KaimingHe/deep-residual-networks). You can do so by
downloading the model and saving it as
`pose/training/config/modelname/init.caffemodel`.


The `pose.py` script will produce a pose prediction for an image. It assumes
that a model with name `pose` has been trained (or downloaded). We normalize the
training images w.r.t. person size, that's why the model works best for images
with a rough person height of 500 pixels. Multiple people are not taken into
account; for every joint the `arg max` position is used over the full image.

### 31 part segmentation

The folder setup is just as for the keypoint estimation: use
`segmentation/tools/create_dataset.py` to create a segmentation dataset from the
UP-3D data or download it (again, we used target person size 500). Then use
`run.sh {train,test,evaluate,trfull,tefull,evfull} modelname` as described above
to create your models. The `segmentation.py` script can be used to get
segmentation results for the model named `segmentation` from and image. We
initialized our models from the Deeplab trained models available
[here](http://liangchiehchen.com/projects/DeepLabv2_resnet.html). Move the
model file to `segmentation/training/modelname/init.caffemodel`.

## Website, citation, license

You can find more information on the [website](http://up.is.tuebingen.mpg.de).
If you use this code for your research, please consider citing us:

```
@inproceedings{Lassner:UP:2017,
  title = {Unite the People: Closing the Loop Between 3D and 2D Human Representations},
  author = {Lassner, Christoph and Romero, Javier and Kiefel, Martin and Bogo, Federica and Black, Michael J. and Gehler, Peter V.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  month = jul,
  year = {2017},
  url = {http://up.is.tuebingen.mpg.de},
  month_numeric = {7}
}
```

License: [Creative Commons Non-Commercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The code for 3D fitting is based on the [SMPLify](http://smplify.is.tue.mpg.de)
code. Parts of the files in the folder `up_tools` (`capsule_ch.py`,
`capsule_man.py`, `max_mixture_prior.py`, `robustifiers.py`,
`sphere_collisions.py`) as well as the model
`models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` fall under the SMPLify
license conditions.
