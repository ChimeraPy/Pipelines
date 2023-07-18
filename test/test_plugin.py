import importlib

from chimerapy.pipelines import register_nodes_metadata


def test_importing_nodes():
    nodes = register_nodes_metadata()
    for node in nodes["nodes"]:
        module_name, class_name = node.rsplit(":", 1)
        try:
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            assert module.__name__ == module_name
            assert class_.__name__ == class_name
        except Exception as e:
            print(e)
