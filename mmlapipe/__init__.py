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


def register_nodes_metadata():
    nodes = {
        "description": "A repository of sharable chimerapy nodes with different functionalities.",
        "nodes": [
            "mmlapipe.generic_nodes.video_nodes:Video",
            "mmlapipe.generic_nodes.video_nodes:ShowWindows",
            "mmlapipe.generic_nodes.log_reader:LogReader",
            "mmlapipe.generic_nodes.screen_capture:ScreenCapture",
            "mmlapipe.mf_sort_tracking.bbox_painter:BBoxPainter",
            "mmlapipe.mf_sort_tracking.detector:MFSortDetector",
            "mmlapipe.mf_sort_tracking.tracker:MFSortTracker",
            "mmlapipe.mf_sort_tracking.video:MFSortVideo",
            "mmlapipe.mf_sort_tracking.anonymizer:Anonymizer",
            "mmlapipe.yolo_node:YOLONode",
            "mmlapipe.pose_node:PoseNode",
            "mmlapipe.save_node:SaveNode",
            "mmlapipe.embodied.gaze:GazeL2CSNet",
            "mmlapipe.embodied.log_processor:GEMSTEPLogProcessor",
        ],
    }

    return nodes
