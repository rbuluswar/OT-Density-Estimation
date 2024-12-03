# OT-Density-Estimation
This is a work-in-progress research project that I am conducting with Prof. Lorenzo Orecchia in the University of Chicago Computer Science department. The goal is to use optimal transport theory to non-parametrically estimate the underlying probability density of a distribution on R^2 with compact support (in this case a square), given a large number of samples. Ultimately, we hope to apply the results to problems like image generation, as an alternative to diffusion models.

This project is written entirely in python, utilizing numpy, OpenCV (https://opencv.org/) to read in images, and Python Optimal Transport (https://pythonot.github.io/) to compute optimal couplings. 
