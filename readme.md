# RADA: Reliability-Aware Domain Adaptation for Time Series Anomaly Detection

This repository contains the source code for the research work RADA, which is currently under submission to the VLDB conference.

RADA is an unsupervised domain adaptation framework for time series anomaly detection built on PyTorch. By introducing innovative mechanisms for reliability filtering and dual-view multi-granularity feature extraction, it effectively addresses the negative transfer problem during cross-domain transfer.

## Introduction

In Time Series Anomaly Detection (TSAD), model training and generalization face significant challenges due to the scarcity of labeled data. Unsupervised Domain Adaptation (UDA) provides a solution by leveraging labeled data from a related source domain to detect anomalies in an unlabeled target domain. However, in practical applications, traditional UDA models often ignore the pre-existing anomalous data in the target domain. Forcibly aligning the distributions of the source domain and the target domain containing anomalies inevitably leads to severe "negative transfer" phenomena.

To overcome this limitation, we propose the RADA framework. RADA innovatively introduces a reliability-aware filtering mechanism. The framework can effectively determine feature reliability, and intercept and filter out potential anomalies in the target domain while maintaining an extremely low computational overhead. Furthermore, to better capture the temporal dependencies of time series, RADA constructs a dual-view multi-granularity feature extraction module. This module can comprehensively extract the deep dynamic features and dependencies of time series at different granularities to facilitate better filtering.

## Installation

To use this code, please follow the steps below to configure your environment:

**Clone this repository:**

```
git clone https://github.com/allzna/RADA.git
```

**Create and activate a virtual environment:**

```
conda create -n rada_env python=3.10
conda activate rada_env
```

**Install required dependencies:**

```
pip install -r requirements.txt
```

## Run the Main UDA Training Loop

For the complete cross-domain adaptation pipeline on a specific dataset, you can directly run the main entry point:

```
python run_uda_loop.py
```