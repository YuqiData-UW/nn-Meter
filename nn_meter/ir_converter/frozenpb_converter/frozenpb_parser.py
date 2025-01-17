# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
import copy
import logging
from tqdm import tqdm
from .protobuf_helper import ProtobufHelper
from nn_meter.utils.import_package import try_import_tensorflow
from ...utils.utils import IS_TF2, DataFormat
from nn_meter.utils.graph_tool import ModelGraph

logging = logging.getLogger("nn-Meter")


class FrozenPbParser:
    def __init__(self, pb_file):
        tf = try_import_tensorflow()
        f = open(pb_file, "rb")
        graph = tf.compat.v1.GraphDef()
        graph.ParseFromString(f.read())

        self.graph = graph

    @staticmethod
    def strip_useless_nodes(model_graph):
        """
        Remove nodes that does not matter with the predict or the structure of model,
        including following types:
        - weights for ops
        - attributes for ops

        Parameters
        ----------
        model_graph : ModelGraph
            the graph holder
        """
        stripped_nodes_type_all = []
        stripped_nodes_type = ["Const", "Identity"]
        stripped_nodes_keywords = [
            "/weight",
            "/weight/read",
            "/ReadVariableOp",
            "/kernel",
            "/gamma",
            "/beta",
            "/moving_mean",
            "/moving_variance",
            "/bias",
            "/reduction_indices",
            "/shape",
            "/split_dim",
            "/axis",
        ]
        stripped_nodes_keywords_tf2 = ["Identity"]
        graph = model_graph.get_graph()
        removed_node = set()
        for key, value in graph.items():
            if "attr" in value.keys():
                if value["attr"]["type"] in stripped_nodes_type:
                    if IS_TF2:
                        stripped_nodes_keywords += stripped_nodes_keywords_tf2
                    for kw in stripped_nodes_keywords:
                        if kw in key:
                            removed_node.add(key)
                            break
                if value["attr"]["type"] in stripped_nodes_type_all:
                    removed_node.add(key)
            elif IS_TF2:
                if key[0] == "^":
                    removed_node.add(key)
        for key in removed_node:
            del graph[key]
        if IS_TF2:
            for key in graph:
                if "inbounds" in graph[key]:
                    graph[key]["inbounds"] = [
                        n for n in graph[key]["inbounds"] if n not in removed_node
                    ]
                if "outbounds" in graph[key]:
                    graph[key]["outbounds"] = [
                        n for n in graph[key]["outbounds"] if n not in removed_node
                    ]

        model_graph.refresh()

    @staticmethod
    def fix_split_naming(model_graph):
        """
        TensorFlow is using "NODE_NAME:NUMBER"  for example "split:0", "split:1"
        as a notation to oredered outputs,
        such notation will make the edge not able to connect. We patch it by using the list
        to store the name and keep the sequence.

        Parameters
        ----------
        model_graph : ModelGraph
            the graph holder
        """
        graph = model_graph.get_graph()
        graph_nodes = copy.deepcopy(list(graph.keys()))
        remove_node_list = []
        for graph_node in graph_nodes:
            if graph_node in graph.keys():
                if "attr" in graph[graph_node].keys():
                    if (
                        graph[graph_node]["attr"]["type"] == "Split"
                        and ":" not in graph_node
                    ):
                        logging.info("Find split main node %s." % graph_node)
                        split_node_name = graph_node
                        for node_name in graph.keys():
                            idx = re.findall(r"%s:(\d+)" %
                                             split_node_name, node_name)
                            if len(idx) > 0:
                                idx = int(idx[0])
                                logging.info(
                                    "Find split child node %s." % node_name)
                                graph[graph_node]["outbounds"] += graph[node_name][
                                    "outbounds"
                                ]
                                graph[graph[node_name]["outbounds"][0]]["inbounds"] += [
                                    graph_node
                                ]
                                remove_node_list.append(node_name)

        for node in remove_node_list:
            del graph[node]

        model_graph.refresh()

    def fetch_attr_to_dict(self, node, is_nchw: bool):
        """
        Tensorflow store some of the attributes as a node connect to the tensor.
        We fetch the attribute from those noed to a dict.

        Parameters
        ----------
        node : Protobuf.node
            The protobuf node of the frozen pb.
        """

        attr_dict = {}

        attr_as_node = {
            "Split": {
                "node_name": lambda x: x + "/split_dim",
                "attr_name": "split_dim",
                "node_value": lambda x: ProtobufHelper.get_tensor_value(x),
            },
            "Mean": {
                "node_name": lambda x: x + "/reduction_indices",
                "attr_name": "reduction_indices",
                "node_value": lambda x: ProtobufHelper.pkg42dec(x.tensor_content),
            },
            "Reshape": {
                "node_name": lambda x: x + "/shape",
                "attr_name": "shape",
                "node_value": lambda x: ProtobufHelper.pkg42dec(x.tensor_content),
            },
            "Concat": {
                "node_name": lambda x: x + "/axis",
                "attr_name": "axis",
                "node_value": lambda x: ProtobufHelper.get_tensor_value(x),
            },
            "ConcatV2": {
                "node_name": lambda x: x + "/axis",
                "attr_name": "axis",
                "node_value": lambda x: ProtobufHelper.get_tensor_value(x),
            },
            "Const": {
                "node_name": lambda x: x,
                "attr_name": "constant",
                "node_value": lambda x: ProtobufHelper.get_tensor_value(x),
            },
            "Pack": {
                "node_name": lambda x: x + r"/(\d)",
                "regex": True,
                "attr_name": "constant",
                "node_value": lambda x: ProtobufHelper.get_tensor_value(x),
            },
        }

        list_i_nodes = ["dilations", "strides", "ksize"]
        str_nodes = ["padding", "data_format"]

        for attr_name in node.attr.keys():
            if attr_name in list_i_nodes:
                attr_dict[attr_name] = [int(a)
                                        for a in node.attr[attr_name].list.i]
                if is_nchw and len(attr_dict[attr_name]) == 4:
                    attr_dict[attr_name] = [attr_dict[attr_name][0], attr_dict[attr_name]
                                            [2], attr_dict[attr_name][3], attr_dict[attr_name][1]]
                continue

            if attr_name in str_nodes:
                attr_dict[attr_name] = node.attr[attr_name].s
                continue

            if attr_name == "value":
                shape = []
                for dim in node.attr[attr_name].tensor.tensor_shape.dim:
                    shape.append(dim.size)
                attr_dict["tensor_shape"] = list(map(int, shape))
                continue

            if attr_name == "shape":
                shape = []
                for dim in node.attr[attr_name].shape.dim:
                    shape.append(dim.size)
                if is_nchw:
                    shape = [shape[0], shape[3], shape[2], shape[1]]
                attr_dict["shape"] = list(map(int, shape))
                continue

        if node.op in attr_as_node.keys():
            # TODO: can use a hash table to reduce complexity
            def check_copy_attr(attr_name):
                copy_attr = False
                if (
                    attr_name == "value"
                    and IS_TF2
                    and "ReadVariableOp" not in node.name
                ):
                    copy_attr = True
                elif (
                    not IS_TF2
                    and attr_name == "value"
                    and "weight" not in node.name
                    and "BatchNorm" not in node.name
                    and "kernel" not in node.name
                ):
                    copy_attr = True
                return copy_attr

            for target_node in self.graph.node:
                if "regex" in attr_as_node[node.op].keys():
                    node_attr = re.findall(
                        attr_as_node[node.op]["node_name"](
                            node.name), target_node.name
                    )
                    if len(node_attr) > 0:
                        logging.info("Find regex matching node %s" % node.name)
                        for attr_name in target_node.attr.keys():
                            if check_copy_attr(attr_name):
                                node_attr_name = attr_as_node[node.op]["attr_name"]
                                if node_attr_name not in attr_dict.keys():
                                    attr_dict[node_attr_name] = []
                                attr_dict[node_attr_name].append(
                                    copy.deepcopy(
                                        attr_as_node[node.op]["node_value"](
                                            target_node.attr[attr_name].tensor
                                        )
                                    )
                                )
                else:
                    if target_node.name == attr_as_node[node.op]["node_name"](
                        node.name
                    ):
                        for attr_name in target_node.attr.keys():
                            if check_copy_attr(attr_name):
                                value = attr_as_node[node.op]["node_value"](
                                    target_node.attr[attr_name].tensor
                                )
                                if (is_nchw and attr_as_node[node.op]["attr_name"] == "constant" and len(value) == 8):
                                    value = value[:2] + value[4:] + value[2:4]
                                elif (is_nchw and node.op == "Mean" and len(value) == 2):
                                    value = [v - 1 for v in value]
                                elif (is_nchw and attr_as_node[node.op]["attr_name"] == "axis"):
                                    value = [v + 2 for v in value]
                                attr_dict[
                                    attr_as_node[node.op]["attr_name"]
                                ] = copy.deepcopy(value)
        return attr_dict

    def check_data_format(self, model_graph: ModelGraph):
        for node in self.graph.node:
            for attr_name in node.attr.keys():
                if attr_name == "data_format" and (node.attr[attr_name].s == DataFormat.NCHW.value or node.attr[attr_name].s == DataFormat.NCHW.value.encode("ascii")):
                    model_graph.set_data_format(DataFormat.NCHW)
                    return

    def parse_graph(self, model_graph):
        """
        Parse a frozen protobuf file from tensorflow to graph IR

        Parameters
        ----------
        model_graph : ModelGraph
            The Graph IR holder.
        """
        valid_node_names = {node.name for node in tqdm(self.graph.node)}
        self.check_data_format(model_graph)
        for node in tqdm(self.graph.node):
            model_graph.node(str(node.name), list(
                map(str, node.input)), valid_node_names)
            model_graph.set_node_attr(
                node.name,
                {
                    "name": str(node.name),
                    "type": str(node.op),
                    "output_shape": [],  # This will be filled later
                    "attr": self.fetch_attr_to_dict(node, model_graph.get_data_format() == DataFormat.NCHW),
                },
            )
