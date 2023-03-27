import pathlib
import os
import sys

# Third-party Imports
import chimerapy as cp
cp.debug([])

# Internal Imports
import mmlapipe

# Constant
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent
DATA_DIR = GIT_ROOT/'data'/'KinectData'
SESSION_VIDEOS = {
    'OELE01': DATA_DIR/'OELE01'/'2022-10-05--11-52-55',
    'OELE02': DATA_DIR/'OELE02'/'2022-10-05--11-48-33',
    'OELE03': DATA_DIR/'OELE03'/'2022-10-05--11-48-21',
    'OELE08': DATA_DIR/'OELE08'/'2022-10-05--11-52-02'
}
assert all([v.exists() for v in SESSION_VIDEOS.values()])

def show_video(manager: cp.Manager):

    graph = cp.Graph()
    node_names = []
    for k, v in SESSION_VIDEOS.items():
        node = mmlapipe.KinectNode(name=k, kinect_data_folder=v, show=True)
        graph.add_node(node)
        node_names.append(node.name)

    mapping = {'local': node_names}
    manager.commit_graph(
        graph=graph,
        mapping=mapping,
        send_packages=[{"name": "mmlapipe", "path": GIT_ROOT/"mmlapipe"}]
    )

def yolo_pipeline(manager: cp.Manager):

    graph = cp.Graph()
    node_names = []
    yolo_node = mmlapipe.YOLONode(name="yolo", classes=['person'])
    graph.add_node(yolo_node)
    node_names.append(yolo_node.name)

    for k, v in SESSION_VIDEOS.items():
        node = mmlapipe.KinectNode(name=k, kinect_data_folder=v)
        graph.add_node(node)
        graph.add_edge(node, yolo_node)
        node_names.append(node.name)

    mapping = {'local': node_names}
    manager.commit_graph(
        graph=graph,
        mapping=mapping,
        send_packages=[{"name": "mmlapipe", "path": GIT_ROOT/"mmlapipe"}]
    )

if __name__ == "__main__":

    # Create default manager and desired graph
    manager = cp.Manager(logdir=GIT_ROOT/"runs", port=0)
    worker = cp.Worker(name="local")
    worker.connect(host=manager.host, port=manager.port)

    # Wait until workers connect
    while True:
        q = input("All workers connected? (Y/n)")
        if q.lower() == "y":
            break
        elif q.lower() == 'q':
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
        elif q.lower() == 'q':
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
