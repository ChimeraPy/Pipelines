import os
import pathlib
import sys

# Third-party Imports
import chimerapy as cp

cp.debug([])

# Internal Imports
import mmlapipe

# Constant
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent
DATA_DIR = GIT_ROOT / "data"
VIDEO_FOLDER = DATA_DIR / "TestData"
assert VIDEO_FOLDER.exists()


def show_video(manager: cp.Manager):

    graph = cp.Graph()
    node_ids = []
    for v_file in VIDEO_FOLDER.iterdir():
        node = mmlapipe.VideoNode(name=v_file.stem, src=v_file, show=True)
        graph.add_node(node)
        node_ids.append(node.id)

    mapping = {"local": node_ids}
    manager.commit_graph(
        graph=graph,
        mapping=mapping,
        send_packages=[{"name": "mmlapipe", "path": GIT_ROOT / "mmlapipe"}],
    )


def yolo_pipeline(manager: cp.Manager):

    graph = cp.Graph()
    node_ids = []
    yolo_node = mmlapipe.YOLONode(name="yolo", classes=["person"])
    graph.add_node(yolo_node)
    node_ids.append(yolo_node.id)

    for i, v_file in enumerate(VIDEO_FOLDER.iterdir()):
        # if i == 3:
        #     break
        node = mmlapipe.VideoNode(name=v_file.stem, src=v_file)
        graph.add_node(node)
        graph.add_edge(node, yolo_node)
        node_ids.append(node.id)

    mapping = {"local": node_ids}
    manager.commit_graph(
        graph=graph,
        mapping=mapping,
        send_packages=[{"name": "mmlapipe", "path": GIT_ROOT / "mmlapipe"}],
    )


if __name__ == "__main__":

    # Create default manager and desired graph
    manager = cp.Manager(logdir=GIT_ROOT / "runs", port=0)
    worker = cp.Worker(name="local", id="local")
    worker.connect(host=manager.host, port=manager.port)

    # Wait until workers connect
    while True:
        q = input("All workers connected? (Y/n)")
        if q.lower() == "y":
            break
        elif q.lower() == "q":
            manager.shutdown()
            worker.shutdown()
            sys.exit(0)

    # Commit the graph
    try:
        # Configure the Manager (testing different setups)
        # show_video(manager)
        yolo_pipeline(manager)
    except Exception as e:
        manager.shutdown()
        worker.shutdown()
        raise e

    # Wail until user stops
    while True:
        q = input("Ready to start? (Y/n)")
        if q.lower() == "y":
            break
        elif q.lower() == "q":
            manager.shutdown()
            worker.shutdown()
            sys.exit(0)

    manager.start()

    # Wail until user stops
    while True:
        q = input("Stop? (Y/n)")
        if q.lower() == "y":
            break

    manager.stop()
    manager.collect()
    manager.shutdown()
