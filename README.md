##### SeaSTAR

<p align="left">
  <img src="/docs/source/_static/images/SeaSTAR_logo.png" width="500">
</p>

This is the package repository for the ESA Earth Explorer 11 candidate mission 'SeaStar'.


## 1. Installation

### 1.1 Download the **seastar_project** repository

Navigate to the latest release `(v1.0)` on the RHS of the root project page and download and unzip the source code.


### 1.2 Create an environment with Anaconda

To run the code in the project you need to install the required Python packages in an environment. To do this we will use **Anaconda**, which can be downloaded [here](https://www.anaconda.com/download/).

Open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command (change directory) to the directory where you have installed the **seastar_project** repository.

Create a new environment named `seastar` with all the required packages and activate this environment by entering the following commands:

```
>>> conda env create --file env/environment.yml
>>> conda activate seastar
```

To confirm that you have successfully activated `seastar`, your terminal command line prompt should now start with `(seatar)`.


## 2. Running the code

### 2.1 Set parameters for your local environment

From the directory containing the **seastar_project** edit the file **seatarx_config.ini** and set the parameters as required e.g. set the path to the  local directories for the SAR data and for writing the results.

### 2.2 Execute the processor

In the terminal window opened in the **seastar_project** directory enter the following command:

```
>>> python master_processor.py
```

## 3. Documentation

[readthedocs](https://seastar-project.readthedocs.io/en/latest/)
[SeaSTAR](/docs/build/html/index.html)
