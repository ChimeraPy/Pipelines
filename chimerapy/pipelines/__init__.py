import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        }
    },
    "loggers": {
        "c2mmla": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}

# Setup the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

__version__ = "0.0.1"


def register_nodes_metadata():
    nodes = {
        "description": "A repository of sharable chimerapy nodes with different functionalities.",
        "nodes": [
            "chimerapy.pipelines.generic_nodes.video_nodes:Video",
            "chimerapy.pipelines.generic_nodes.video_nodes:ShowWindows",
            "chimerapy.pipelines.generic_nodes.log_reader:LogReader",
            "chimerapy.pipelines.generic_nodes.screen_capture:ScreenCapture",
            "chimerapy.pipelines.generic_nodes.audio_node:AudioNode",
            "chimerapy.pipelines.mf_sort_tracking.bbox_painter:BBoxPainter",
            "chimerapy.pipelines.mf_sort_tracking.detector:MFSortDetector",
            "chimerapy.pipelines.mf_sort_tracking.tracker:MFSortTracker",
            "chimerapy.pipelines.mf_sort_tracking.video:MFSortVideo",
            "chimerapy.pipelines.mf_sort_tracking.anonymizer:Anonymizer",
            "chimerapy.pipelines.yolo_node:YOLONode",
            "chimerapy.pipelines.embodied.gaze:GazeL2CSNet",
            "chimerapy.pipelines.embodied.log_processor:GEMSTEPLogProcessor",
            "chimerapy.pipelines.pose.video:YOLOVideo",
            "chimerapy.pipelines.pose.multi_vid_pose:MultiPoseNode",
            "chimerapy.pipelines.pose.multi_save:MultiSaveNode",
            "chimerapy.pipelines.pose.display:DisplayNode",
        ],
    }

    return nodes
