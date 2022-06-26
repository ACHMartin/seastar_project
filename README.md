##### Seastar

<p align="left">
  <img src="/docs/images/seastar_img3.jpg" width="500">
</p>

general project info can go here


## 1. Installation

### 1.1 Download the **seastar_project** repository

Navigate to the latest release on the RHS of this page and download and unzip the source code.


### 1.2 Create an environment with Anaconda

To run the code in the project you need to install the required Python packages in an environment. To do this we will use **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/).

Open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command (change directory) to go the folder where you have downloaded the **seastar_project** repository.

Create a new environment named `seastar` with all the required packages and activate this environment by entering the following commands:

```
>>> conda create --file env/environment.yml
>>> conda activate seastar
```

To confirm that you have successfully activated `seastar`, your terminal command line prompt should now start with `(seatar)`.


## 2. Running the code

### 2.1 Set parameters for your local environment

From the directory containing the **seastar_project** edit the file **seatarx_config.ini** and set the parameters as required e.g. set the path to the  local directories for the SAR data and for writing the results.

### 2.2 Execute the processor

In the terminal window opened in the **seastar_project** directory enter the follwing command:

```
>>> python master_processor.py
```

