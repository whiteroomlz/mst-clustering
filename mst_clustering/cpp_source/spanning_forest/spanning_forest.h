#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

class SpanningForest {
public:
    struct Edge {
        int32_t first_node;
        int32_t second_node;
        double edge_weight;

        Edge(int32_t first_node, int32_t second_node, double edge_weight)
            : first_node(first_node), second_node(second_node), edge_weight(edge_weight) {
        }
    };

    SpanningForest(const size_t size)
        : roots_(std::vector<int32_t>(size)),
          dsu_weights_(std::vector<size_t>(size, 1)),
          trees_count_(size) {
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
