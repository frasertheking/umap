<div align="center">

![logo](https://github.com/frasertheking/umap/blob/main/images/banner.png?raw=true)

Decoding Nonlinear Signals in Multidimensional Precipitation Observations, maintained by [Fraser King](https://frasertheking.com/)

</div>

## Overview

This is the public code repository for our research article.

Changes in the phase of precipitation reaching the surface has far-reaching implications to agricultural productivity, fresh water availability, outdoor recreation economies, and ecosystem sustainability. In this study, we aimed to improve precipitation data analysis for weather prediction by examining over 1.5 million minute-scale particle measurements from seven sites over ten years. Using nonlinear dimensionality reduction techniques, we reduced the data complexity by 75% and identified nine unique precipitation groups. The nonlinear technique provided clearer separation than traditional linear methods, with fewer ambiguous cases and better categorization of important hydrometeor properties like precipitation phase and intensity. These findings enhance our understanding of global precipitation patterns by revealing hidden features in large, complex datasets.

![overview](https://github.com/frasertheking/umap/blob/main/images/overview.png?raw=true)

This repository contains the processing and analysis scripts used in the article, figure plotting code and an example interactive notebook for experimenting with some of the precipitation data yourself using similar techniques. The goal of this repository is to provide open access to other for reproducing our results, or adapting them for future work.

## UMAP+HDBSCAN Manifold

To play with the data yourself, please see our [interactive tool](https://frasertheking.com/interactive/). You can see an example of what the UMAP+HDBSCAN precipitation clusters look like in the animated image below.

<p align="center">
    <img src="https://github.com/frasertheking/umap/blob/main/images/animated.gif?raw=true" />
</p>

## Lookup-Table Utilities (`lut_tools.py`)

<div align="center">

<img src="https://github.com/frasertheking/umap/blob/main/images/lut_map_demo.png?raw=true"
     alt="Cluster lookup-table visualisation" width="80%" />

</div>

### What is this?

The **UMAP + HDBSCAN** analysis in this repo collapses 1.5 million five-minute
hydrometeor snapshots into **nine physically meaningful precipitation regimes**.
We distilled those results into a *3-D* histogram with a 10-element probability 
vector in every cell \\(P(\\text{cluster}\\;|\\;T,\\;\\log_{10}N_t)\\), 
where each bin stores the probability of observing a given cluster at surface 
temperature *T* and total particle concentration *N_t*.

These probabilities are **valuable priors**:

* **Bayesian retrievals** can make use of this prior using two simple atmospheric
  variables to improve the predictive accuracy of the retrieved particle phase
* **Numerical model parameterisations** can sample from the conditional
  distribution to initialise microphysics schemes or to regularise forecasts

### LUT Quick-start
We have worked to make this easy to implement in other models:

```bash
pip install pandas numpy pyarrow matplotlib
```

```python
from lut_tools import load_lookup_table, query_lookup, plot_lut_map

# 1 Load the parquet LUT (defaults to ./umap_cluster_prior.parquet)
lut = load_lookup_table()

# 2 Query any (T, Nt) point – returns a Series summing to 1.0
probs = query_lookup(lut, T=-2.0, Nt_log=3.0)
print(probs.sort_values(ascending=False)[:3])
# 0.45 → Heavy Snow  |  0.32 → Heavy Snow→Mixed  |  0.16 → Light Snow  …

# 3 Visualise or regenerate the map
plot_lut_map(lut, "lut_map_demo.png")
```

## Data Sources

A Comprehensive Northern Hemisphere Particle Microphysics Dataset from the Precipitation Imaging Package

The data for this project is hosted online on UM's [DeepBlue repository](https://deepblue.lib.umich.edu/data/concern/data_sets/kk91fm40r?locale=en).

We have collected PIP microphysical data from a variety of measurement locations across the northern hemisphere. Data originally in a proprietary ASCII format has been converted to the more universally recognized NetCDF-4 format for ease of sharing and compatibility within the academic community. The conversion process, undertaken using a combination of bash and Python, ensures broader compatibility with various data analysis tools and platforms. A quality assurance (QA) procedure has been undertaken to ensure the integrity of the data. Post QA, the data is transformed into daily NetCDF-4 files following the Climate and Forecast (CF) conventions (version 1.10) and compressed with a level 2 deflation for optimized file size. Additional details into the data curation process can be found in our journal article publication.

We have also built a custom API for interacting with the PIP data called **pipdb**. For information on how to use the API please see our [readthedocs documentation](https://pipdb.readthedocs.io/en/latest/).

## Previous Work

To see how we previously used PCA to identify modes of snowfall variaiblity, please see our associated [GitHub repository](https://github.com/frasertheking/snowfall_pca).

![pca](https://github.com/frasertheking/umap/blob/main/images/pca.png?raw=true)

## Installation

    git clone https://github.com/frasertheking/umap.git
    conda env create -f env.yml
    conda activate umap

## Examples

We also provide an interactive Google Colab environment to experiment with (and for reproducing our results), with a subsample of our full dataset. To view the notebook please click the following button:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bESVTHSmwZEdv5MyZQIRvarMJIhmWOMF?usp=share_link)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Note that, as a living project, code is not as clean as it could (should) be, and unit tests need to be produced in future iterations to maintain stability.

## Authors & Contact

- Fraser King, University of Michigan, kingfr@umich.edu
- Claire Pettersen, University of Michigan
- Brenda Dolan, Colorado State University
- Julia Shates, NASA Jet Propulsion Laboratory
- Derek Posselt, NASA Jet Propulsion Laboratory

![logos](https://github.com/frasertheking/umap/blob/main/images/logos.png?raw=true)

## Funding
This project was primarily funded by NASA New (Early Career) Investigator Program (NIP) grant at the [University of Michigan](https://umich.edu). The Natural Sciences and Engineering Research Council of Canada (NSERC) also provided funding via a PDF award.
