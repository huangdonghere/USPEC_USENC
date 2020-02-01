# Ultra-Scalable Spectral Clustering and Ensemble Clustering

## Overview

This repository provides the Matlab source code for two large-scale clustering algorithms, namely, `Ultra-Scalable Spectral Clustering (U-SPEC)` and `Ultra-Scalable Ensemble Clustering (U-SENC)`, both of which have nearly linear time and space complexity and are capable of robustly and efficiently partitioning ten-million-level nonlinearly-separable datasets on a PC with 64GB memory.

If you find this repository helpful for your research, please cite the paper below. 

```
Dong Huang, Chang-Dong Wang, Jian-Sheng Wu, Jianhuang Lai, and Chee-Keong Kwoh.
Ultra-Scalable Spectral Clustering and Ensemble Clustering, 
IEEE Transactions on Knowledge and Data Engineering (TKDE), in press, 2020. 
DOI: https://doi.org/10.1109/TKDE.2019.2903410
```

## Description of Files

### Code

|Function | Description |
| ----------------- | :----------------: |
|`demo_1_USPEC.m` | A demo of the U-SPEC algorithm.|
|`demo_2_USENC.m` | A demo of the U-SENC algorithm.|
|`USPEC.m` | Call this function to perform the U-SPEC algorithm.|
|`USENC.m` | Call this function to perform the U-SENC algorithm.|
|`litekmeans.m`| A fast implementation of k-means. |
|`computeNMI.m`| Call this function to compute the NMI score.|
|`synthesizeLargescaleDatasets.p`| Call this function to produce the five large-scale synthetic datasets, whose sizes range from one million to twenty million. |
|`synthesizeLargescaleDatasets_withArbitrarySizes.p`| Produce the five synthetic datasets with arbitrary sizes.|

### Data

In this repository, we provide the files of the five real-world datasets, namely, PenDigits, USPS, Letters, MNIST, and Covertype. We also provide the MATLAB code (`synthesizeLargescaleDatasets.p`) to reproduce all of the five large-scale synthetic datasets used in our paper.

## Any Questions

Don't hesitate to contact me if you have any questions regarding this work. (Email: huangdonghere at gmail dot com)
