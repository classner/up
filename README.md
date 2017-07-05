# Unite the People code repository

Requirements:

* OpenCV (on Ubuntu, e.g., install libopencv-dev and python-opencv).
* SMPL (download at http://smpl.is.tue.mpg.de/downloads) and unzip to a 
  place of your choice.
* OpenDR (just run `pip install opendr`, unfortunately can't be done automatically
  with the setuptools requirements.
* If you want to train a segmentation model, Deeplab V2
  (https://bitbucket.org/aquariusjay/deeplab-public-ver2) with a minimal patch
  applied that can be found in the subdirectory `patches`, to enable on the fly
  mirroring of the segmented images. Since I didn't use the MATLAB interface and
  did not care about fixing related errors, I just deleted 
  `src/caffe/layers/mat_{read,write}_layer.cpp` as well as
  `src/caffe/util/matio_io.cpp` and built with `-DWITH_matlab=Off`.
* If you want to train a pose model, the Deepercut caffe
  (https://github.com/eldar/deepcut-cnn).
* If you want to get deepercut-cnn predictions, download the deepercut .caffemodel
  file and place it in models/pose/deepercut.caffemodel.
* Edit the file `config.py` to set up the paths.

The rest of the requirements is then automatically installed when running:

```
python setup.py develop
```

You can find more information on the [website](http://up.is.tuebingen.mpg.de). If
you use this code for your research, please consider citing us:

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
