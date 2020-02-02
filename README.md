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
|`synthesizeLargescaleDatasets.p`| Call this function to synthesize the five large-scale datasets, whose sizes range from one million to twenty million. |
|`synthesizeLargescaleDatasets_withArbitrarySizes.p`| Produce the five synthetic datasets with arbitrary sizes.|

### Data

In this repository, we provide the files of the five real-world datasets, namely, PenDigits, USPS, Letters, MNIST, and Covertype. We also provide the MATLAB code to reproduce the five large-scale synthetic datasets used in our paper.

#### How to Reproduce the Synthetic Datasets?

- To generate the five large-scale synthetic datasets, you can call the `synthesizeLargescaleDatasets` function, which has just one input parameter. Note that this input parameter can be set to one of the five data names:

	* 'TB1M' 
	* 'SF2M' 
	* 'CC5M' 
	* 'CG10M' 
	* 'Flower20M'
  
  Example (to synthesize the CC5M dataset):
  
  ```
  synthesizeLargescaleDatasets('CC5M');
  % The synthesized dataset will be saved in 'data_CC5M.mat'.
  ```
- To generate the five synthetic datasets with arbitrary sizes, you can call the `synthesizeLargescaleDatasets_withArbitrarySizes` function, which has two input parameters, that is
	* dataName: can be one of the five names: 	'TB', 	'SF', 	'CC', 	'CG', 	'Flower'
	* dataSize: can be set to any integers, provided that you have enough space to save the data.
  
  Example (to synthesize a CG dataset with one million points):
  
  ```
  dataName = 'CG';
  dataSize = 1000000;
  synthesizeLargescaleDatasets_withArbitrarySizes(dataName, dataSize); 
  % The synthesized dataset will be saved in 'data_CG_1000000.mat'.
  ```

## Further Questions?

Don't hesitate to contact me if you have any questions regarding this work. (Email: huangdonghere at gmail dot com)
