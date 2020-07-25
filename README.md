# mirt-demo

 https://github.com/JeffFessler/mirt-demo

Collection of
Jupyter notebooks that demonstrate the image reconstruction capabilities
of
[MIRT.jl](https://github.com/JeffFessler/MIRT.jl),
the Julia version of the Michigan Image Reconstruction Toolbox.

You can test drive any of these Jupyter notebooks
in a browser
without installing any local software
by using the free service at
https://mybinder.org/
by clicking on the binder badges below:

* Basic introduction:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeffFessler/mirt-demo/master?filepath=isbi-19%2F00-intro.ipynb)

* MRI compressed sensing reconstruction demo:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeffFessler/mirt-demo/master?filepath=isbi-19%2F01-recon.ipynb)

* Dynamic MRI with golden-angle radial sampling reconstruction demo:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeffFessler/mirt-demo/master?filepath=mri%2Fmri-sim-2d%2Bt.ipynb)

You can also view the notebook code directly:
* [isbi-19 demos](https://github.com/JeffFessler/mirt-demo/blob/master/isbi-19/)
* [other MRI demos](https://github.com/JeffFessler/mirt-demo/blob/master/mri/)

Here is an example of the kind of image produced by one of the demos
![phantom-image](/figure/isbi-19-recon1.png?raw=true "phantom image")


### Notes

The `Makefile` uses some very useful command-line tools,
namely
[jupytext](https://github.com/mwouts/jupytext)
[nbconvert](https://nbconvert.readthedocs.io)
to automatically
create the Jupyter notebooks
and a corresponding html version
from the Julia source code.

This software was developed at the University of Michigan 
by 
[Jeff Fessler](http://web.eecs.umich.edu/~fessler)
and his 
[group](http://web.eecs.umich.edu/~fessler/group).
