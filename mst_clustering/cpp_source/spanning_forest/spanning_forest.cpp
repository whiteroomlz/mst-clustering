#pragma once

#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <utility>
#include <numeric>
#include <memory>
#include <vector>
#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class SpanningForest {
public:
    struct Edge {
        int32_t first_node; int32_t second_node; double edge_weight;

        Edge(int32_t first_node, int32_t second_node, double edge_weight)
            : first_node(first_node)
            , second_node(second_node)
            , edge_weight(edge_weight) {}
    };

    SpanningForest(const size_t size)
        : roots_(std::vector<int32_t>(size))
        , dsu_weights_(std::vector<size_t>(size, 1))
        , trees_count_(size) {
        std::iota(roots_.begin(), roots_.end(), 0);
    }

    size_t size() const;

    bool isSpanningTree() const;

    int32_t findRoot(const int32_t item);

    void getRoots(py::list result);

    void getEdges(const int32_t root, py::list result);

    void addEdge(const int32_t first_node, const int32_t second_node, const double edge_weight);

    void removeEdge(int32_t first_node, int32_t second_node);

private:
    std::unordered_multimap<int32_t, std::shared_ptr<Edge>> edges_;

    std::vector<int32_t> roots_;

    std::vector<size_t> dsu_weights_;

    size_t trees_count_;

private:
    void dsuUnite(const int32_t first_node, const int32_t second_node);

    void getTreeItems(int32_t node, std::unordered_set<int32_t>* unique_nodes,
        std::unordered_set<std::shared_ptr<Edge>>* edges) const;
};

PYBIND11_MODULE(spanning_forest, m) {
    py::class_<SpanningForest::Edge>(m, "Edge")
        .def(py::init<int32_t, int32_t, double>(), py::arg("parent_id"), py::arg("child_id"), py::arg("weight"))
        .def_readonly("first_node", &SpanningForest::Edge::first_node)
        .def_readonly("second_node", &SpanningForest::Edge::second_node)
        .def_readonly("edge_weight", &SpanningForest::Edge::edge_weight);

    py::class_<SpanningForest>(m, "SpanningForest")
        .def(py::init<size_t>(), py::arg("capacity"))
        .def("size", &SpanningForest::size, "Returns the number of spanning trees in the forest.")
        .def("is_spanning_tree", &SpanningForest::isSpanningTree, "Returns true if the forest contains only one root.")
        .def("find_root", &SpanningForest::findRoot, "Returns the root of the specified node.", py::arg("node"))
        .def("get_roots", &SpanningForest::getRoots, "Returns all roots ids.", py::arg("output_list"))
        .def("get_edges", &SpanningForest::getEdges, "Returns all edges of root with the specified id.",
            py::arg("root"), py::arg("output_list"))
        .def("add_edge", &SpanningForest::addEdge, "Adds an edge between the two nodes with specified ids.",
            py::arg("first_node"), py::arg("second_node"), py::arg("edge_weight"))
        .def("remove_edge", &SpanningForest::removeEdge, "Removes the edge between the two nodes with specified ids.",
            py::arg("first_node"), py::arg("second_node"));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

size_t SpanningForest::size() const {
    return trees_count_;
}

bool SpanningForest::isSpanningTree() const {
    return trees_count_ == 1;
}

int32_t SpanningForest::findRoot(const int32_t node) {
    if (node == roots_[node]) {
        return node;
    }

    return roots_[node] = findRoot(roots_[node]);
}

void SpanningForest::getRoots(py::list result) {
    std::set<int32_t> roots;

    for (int32_t node = 0; node < roots_.size(); ++node) {
        roots.insert(findRoot(node));
    }

    for (int32_t root : roots) {
        result.append(root);
    }
}

void SpanningForest::getEdges(const int32_t root, py::list result) {
    std::unordered_set<int32_t> second_tree_nodes;
    std::unordered_set<std::shared_ptr<Edge>> edges;
    getTreeItems(root, &second_tree_nodes, &edges);

    for (std::shared_ptr<Edge> edge : edges) {
        result.append(Edge(edge.get()->first_node, edge.get()->second_node, edge.get()->edge_weight));
    }
}

void SpanningForest::addEdge(const int32_t first_node, const int32_t second_node, const double edge_weight) {
    dsuUnite(first_node, second_node);

    std::shared_ptr<Edge> edge(new Edge{ first_node, second_node, edge_weight });
    edges_.emplace(first_node, edge);
    edges_.emplace(second_node, edge);

    --trees_count_;
}

void SpanningForest::removeEdge(int32_t first_node, int32_t second_node) {
    auto first_values_range = edges_.equal_range(first_node);
    for (auto key_value = first_values_range.first; key_value != first_values_range.second; ++key_value) {
        if ((*key_value).second.get()->second_node == second_node) {
            edges_.erase(key_value);
            break;
        }
    }

    auto second_values_range = edges_.equal_range(second_node);
    for (auto key_value = second_values_range.first; key_value != second_values_range.second; ++key_value) {
        if ((*key_value).second.get()->first_node == first_node) {
            edges_.erase(key_value);
            break;
        }
    }

    std::unordered_set<int32_t> second_tree_nodes;
    std::unordered_set<std::shared_ptr<Edge>> edges;

    getTreeItems(second_node, &second_tree_nodes, &edges);
    if (second_tree_nodes.find(findRoot(second_node)) != second_tree_nodes.end()) {
        std::swap(first_node, second_node);
        second_tree_nodes.clear();
        edges.clear();
        getTreeItems(second_node, &second_tree_nodes, &edges);
    }

    dsu_weights_[findRoot(first_node)] -= second_tree_nodes.size();

    for (const int32_t node : second_tree_nodes) {
        roots_[node] = node;
        dsu_weights_[node] = 1;
    }

    for (const std::shared_ptr<Edge> edge : edges) {
        dsuUnite(edge.get()->first_node, edge.get()->second_node);
    }

    ++trees_count_;
}

void SpanningForest::dsuUnite(const int32_t first_node, const int32_t second_node) {
    int32_t first_parent = findRoot(first_node);
    int32_t second_parent = findRoot(second_node);

    if (first_parent == second_parent) {
        std::stringstream message;
        message << "Uniting the passed nodes {" << first_node << "} and {"
            << second_node << "} will produce a cycle.";
        throw std::invalid_argument(message.str());
    }

    if (dsu_weights_[first_parent] < dsu_weights_[second_parent]) {
        std::swap(first_parent, second_parent);
    }

    roots_[second_parent] = first_parent;
    dsu_weights_[first_parent] += dsu_weights_[second_parent];
}

void SpanningForest::getTreeItems(int32_t node, std::unordered_set<int32_t>* unique_nodes,
    std::unordered_set<std::shared_ptr<Edge>>* edges) const {
    unique_nodes->insert(node);

    auto key_values_range = edges_.equal_range(node);
    for (auto key_value = key_values_range.first; key_value != key_values_range.second; ++key_value) {
        std::shared_ptr<Edge> edge = (*key_value).second;
        edges->insert(edge);

        int32_t other_node = edge->first_node == node ? edge->second_node : edge->first_node;

        if (unique_nodes->find(other_node) == unique_nodes->end()) {
            getTreeItems(other_node, unique_nodes, edges);
        }
    }
}
