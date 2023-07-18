# chimerapy-pipelines
This is a repository of sharable [`ChimeraPy`](https://github.com/ChimeraPy) pipelines, with various Node implementations and [`Orchestrator`](https://github.com/ChimeraPy/Orchestrator) configurations.


## Installation
For a basic installation, clone the repository and install the requirements:

```bash
$ git clone https://github.com/ChimeraPy/Pipelines.git
$ cd Pipelines
$ pip install .
```

## Pipelines
This repository contains several sub-projects with specific dependencies.


### mf_sort
The [`mf_sort_tracking`](chimerapy/pipelines/mf_sort_tracking) package provides an integration for the [`MF-SORT`](https://github.com/kbvatral/MF-SORT) tracking algorithm with `ChimeraPy`.

To install the dependencies for this package, run the following command:

```bash
$ pip install ".[mfsort]"
```

Example configurations for this package are available in the [`configs/mf_sort`](./configs/mf_sort) directory. To run an example configuration out of the box, run the following command:

```bash
$ cp-orchestrator orchestrate --config configs/mf_sort/single_tracker_local_http.json
```

Modify the configurations as needed to run on your system.
