# chimerapy-pipelines
<p align="center">
  <a href="https://github.com/ChimeraPy"><img src="./docs/images/banner.png" alt="ChimeraPy"></a>
</p>
<p align="center">
    <em>Repository of shareable ChimeraPy pipelines</em>
</p>
<p align="center">
</p>


ChimeraPy is a Scientific, Distributed Computing Framework for Real-time Multimodal Data Retrieval and Processing. This is a repository of sharable [`ChimeraPy`](https://github.com/ChimeraPy) pipelines, with various Node implementations and [`Orchestrator`](https://github.com/ChimeraPy/Orchestrator) configurations.


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


### embodied
The [`embodied`](chimerapy/pipelines/mf_sort_tracking) package provides an integration for the [`EmbodiedLearningProject`](https://github.com/oele-isis-vanderbilt/EmbodiedLearningProject) with `ChimeraPy`.

To install the dependencies for this package, run the following command:

```bash
$ pip install ".[embodied]"
```

Example configurations for this package are available in the [`configs/embodied`](./configs/embodied) directory. To run an example configuration out of the box, run the following command:

```bash
$ cp-orchestrator orchestrate --config configs/embodied/gaze_processing.json
```

Modify the configurations as needed to run on your system.

### yolov8

The [`yolov8`](chimerapy/pipelines/yolov8) package provides an integration for the [`YoloV8`](https://github.com/ultralytics/ultralytics) with `ChimeraPy`.

To install the dependencies for this package, run the following command:

```bash
$ pip install ".[yolov8]"
```

Example configurations for this package are available in the [`configs/pose`](./configs/pose) directory. To run an example configuration out of the box, run the following command:

```bash
$ cp-orchestrator orchestrate --config configs/pose/multi_pose_demo.json
```

## Contributing
Contributions are welcomed! Our [Developer Documentation](https://chimerapy.readthedocs.io/en/latest/developer/index.html) should provide more details in how ChimeraPy works and what is in current development.

## License
[ChimeraPy](https://github.com/ChimeraPy) and [ChimeraPy/Pipelines](https://github.com/ChimeraPy/Pipelines) uses the GNU GENERAL PUBLIC LICENSE, as found in [LICENSE](./LICENSE) file.

## Funding Info
This project is supported by the [National Science Foundation](https://www.nsf.gov/) under AI Institute  Grant No. [DRL-2112635](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2112635&HistoricalAwards=false).
