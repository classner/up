# 3D Fitting to 14 or 91 Keypoints

This subfolder contains scripts to fit a 3D body to 14 or 91 keypoints with the
SMPLify fitting objective and render it. The silhouette fitting objective is
currently not included due to licensing issues. The code is based on the
[SMPLify code](http://smplify.is.tuebingen.mpg.de) and some of its library files
in the `up_tools` folder as well as the model
`models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` fall under the SMPLify
license. If you find this code useful for your research, please consider also
citing

```
@inproceedings{Bogo:ECCV:2016,
  title = {Keep it {SMPL}: Automatic Estimation of {3D} Human Pose and Shape from a Single Image},
  author = {Bogo, Federica and Kanazawa, Angjoo and Lassner, Christoph and Gehler, Peter and Romero, Javier and Black, Michael J.},
  booktitle = {Computer Vision -- ECCV 2016},
  series = {Lecture Notes in Computer Science},
  publisher = {Springer International Publishing},
  month = oct,
  year = {2016}
}
```
