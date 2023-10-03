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
            "chimerapy.pipelines.yolov8.video:YOLOVideo",
            "chimerapy.pipelines.yolov8.multi_vid_pose:YoloV8Node",
            "chimerapy.pipelines.yolov8.multi_save:MultiSaveNode",
            "chimerapy.pipelines.yolov8.display:DisplayNode",
            "chimerapy.pipelines.g3.node:G3",
        ],
    }

    return nodes
