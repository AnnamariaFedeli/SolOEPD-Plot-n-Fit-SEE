# SolOEPD-Plot-n-Fit-SEE
 This repository contains a software that enables the user to plot and fit SolO EPD SEE spectra. 

## Installation 
1. Make sure you have a recent version of [Anaconda/conda/miniconda](https://www.anaconda.com/products/distribution) installed. 
2. Open your terminal/command line/Anaconda prompt and run the following:
    ``` bash
    $ conda create --name solosee python=3.9
    $ conda activate solosee
    ```
3. [Download this file](https://github.com/jgieseler/SolOEPD-Plot-n-Fit-SEE/archive/refs/heads/main.zip) and extract to a folder of your choice (or clone the repository https://github.com/jgieseler/SolOEPD-Plot-n-Fit-SEE if you know how to use `git`).
6. Open your terminal/command line/Anaconda prompt, navigate to the downloaded (extracted) folder `SolOEPD-Plot-n-Fit-SEE` (or `SolOEPD-Plot-n-Fit-SEE-main`) that contains the file `requirements.txt`, and run the following (first command is just to verify that you are in the correct conda environment):
    ``` bash
    $ conda activate solosee
    $ pip install -r requirements.txt
    ```


## Run 
1. Open your terminal/command line/Anaconda prompt.
2. In the terminal, navigate to the downloaded (extracted) folder `SolOEPD-Plot-n-Fit-SEE` (or `SolOEPD-Plot-n-Fit-SEE-main`) that contains some `.ipynb` files.
3. Make sure the corresponding conda environment is activated by running `conda activate solosee` in the terminal.
4. Run `jupyter notebook`
5. Your standard web-browser should now open the Jupyter interface, where you can double click on the corresponding `.ipynb` files to launch them.
