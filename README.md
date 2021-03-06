# Machine-Learning Repository
Jupyter Notebooks and some Python code related to machine learning 
* [Bayesian-Linear-Regression](https://github.com/kaibrach/machine-learning/tree/master/bayesian-linear-regression)
* [Gaussian-Processes](https://github.com/kaibrach/machine-learning/blob/master/gaussian-processes)
* [Parameter-Estimation-for-Linear-Regression](https://github.com/kaibrach/machine-learning/tree/master/regression-parameter-estimation)

# Run with Binder
A Binder-compatible repo with an `environment.yml` file.

Binder will search for a dependency file, such as `requirements.txt` or `environment.yml`, 
in the repository's root directory . 
The dependency files will be used to build a Docker image for the notebook. 
If an image has already been built for the given repository, it will not be rebuilt. If a new commit has been made, the image will automatically be rebuilt. 

**Notes**

The environment.yml file should list all Python libraries on which your notebooks depend, specified as though they were created using the following conda commands:

    source activate example-environment
    conda env export --no-builds -f environment.yml

Note that the only libraries available to you will be the ones specified in the `environment.yml`, so be sure to include everything that you need!

Also note that conda will possibly try to include OS-specific packages in `environment.yml`, so you may have to manually prune `environment.yml` to get rid of these packages. 
Confirmed Mac-OSX-specific packages that should be removed are:

    libcxxabi=4.0.1
    appnope=0.1.0
    libgfortran=3.0.1
    libcxx=4.0.1

Confirmed Windows-specific packages that should be removed are:

    m2w64-gcc-libgfortran=5.3.0
    m2w64-gcc-libs=5.3.0
    m2w64-gcc-libs-core=5.3.0
    m2w64-gmp=6.1.0
    m2w64-libwinpthread-git=5.0.0.4634.697f757
    msys2-conda-epoch=20160418
    pywinpty=0.5.5
    vc=14
    vs2015_runtime=14.0.25420
    wincertstore=0.2
    winpty=0.4.3
