<div align="center">

![logo](https://github.com/frasertheking/umap/blob/main/images/banner.png?raw=true)

Decoding Nonlinear Signals in Multidimensional Precipitation Observations, maintained by [Fraser King](https://frasertheking.com/)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

## Overview

This is the public code repository for our research article currently submitted to [Nature Communications](https://www.nature.com/ncomms/).

Changes in the phase of precipitation reaching the surface has far-reaching implications to agricultural productivity, fresh water availability, outdoor recreation economies, and ecosystem sustainability. In this study, we aimed to improve precipitation data analysis for weather prediction by examining over 1.5 million minute-scale particle measurements from seven sites over ten years. Using nonlinear dimensionality reduction techniques, we reduced the data complexity by 75% and identified nine unique precipitation groups. The nonlinear technique provided clearer separation than traditional linear methods, with fewer ambiguous cases and better categorization of important hydrometeor properties like precipitation phase and intensity. These findings enhance our understanding of global precipitation patterns by revealing hidden features in large, complex datasets.

This repository contains the processing and analysis scripts used in the article, figure plotting code and an example interactive notebook for experimenting with some of the precipitation data yourself using similar techniques. The goal of this repository is to provide open access to other for reproducing our results, or adapting them for future work.
![PCA](https://github.com/frasertheking/umap/blob/main/images/animated.gif?raw=true)

<!-- 
---

## Data Sources

A Comprehensive Northern Hemisphere Particle Microphysics Dataset from the Precipitation Imaging Package

The data for this project is hosted online on UM's [DeepBlue repository](https://deepblue.lib.umich.edu/data/concern/data_sets/kk91fm40r?locale=en).

We have collected PIP microphysical data from a variety of measurement locations across the northern hemisphere. Data originally in a proprietary ASCII format has been converted to the more universally recognized NetCDF-4 format for ease of sharing and compatibility within the academic community. The conversion process, undertaken using a combination of bash and Python, ensures broader compatibility with various data analysis tools and platforms. A quality assurance (QA) procedure has been undertaken to ensure the integrity of the data. Post QA, the data is transformed into daily NetCDF-4 files following the Climate and Forecast (CF) conventions (version 1.10) and compressed with a level 2 deflation for optimized file size. Additional details into the data curation process can be found in our journal article publication.

For a brief overview of the data study sites and coverage periods, please see the figure below.

![data overview](https://github.com/frasertheking/snowfall_pca/blob/main/images/fig01.png?raw=true)

## Installation

To perform your own training tests, you'll want to clone this repo, and create a Conda environment with scikit-learn installed. Note that plotly is required for the interactive 3D plots.

## Examples

To interact with the processing scripts, please see the commented Python Jupyter notebooks in the /notebooks path of this repo.

A subset example dataset can also be downloaded [here](https://www.frasertheking.com/downloads/pip_snow_obs_10_sites_500_subsample.csv) in csv format.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Note that, as a living project, code is not as clean as it could (should) be, and unit tests need to be produced in future iterations to maintain stability.

## Authors & Contact

- Fraser King, University of Michigan, kingfr@umich.edu
- Claire Pettersen, University of Michigan
- Brenda Dolan, Colorado State University
- Julia Shates, NASA Jet Propulsion Laboratory
- Derek Posselt, NASA Jet Propulsion Laboratory

## Funding
This project was primarily funded by NASA New (Early Career) Investigator Program (NIP) grant at the [University of Michigan](https://umich.edu).
 -->
