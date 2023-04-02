#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

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

        std::tuple<int32_t, int32_t> nodes() {
            return {first_node, second_node};
        }
    };

    explicit SpanningForest(const size_t size)
        : dsu_roots_(std::vector<int32_t>(size)),
          dsu_weights_(std::vector<size_t>(size, 1)),
          trees_count_(size) {
        for (int node = 0; node < size; ++node) {
            trees_roots_.insert(node);
            dsu_roots_[node] = node;
        }
    }

    size_t size() const;

    bool isSpanningTree() const;

    int32_t findRoot(int32_t node);

    size_t getTreeSize(int32_t root) const;

    void getRoots(std::vector<int32_t>& out_roots) const;

    py::list getTreeInfo(int32_t root);

    py::array_t<int32_t> getTreeNodes(int32_t root);

    std::vector<std::shared_ptr<Edge>> getTreeEdges(int32_t root);

    void addEdge(int32_t first_node, int32_t second_node, double edge_weight);

    void removeEdge(int32_t first_node, int32_t second_node);

private:
    std::unordered_multimap<int32_t, std::shared_ptr<Edge>> edges_;

    std::unordered_set<int32_t> trees_roots_;

    std::vector<int32_t> dsu_roots_;

    std::vector<size_t> dsu_weights_;

    size_t trees_count_;

private:
    void dsuUnite(int32_t first_node, int32_t second_node);

    void traverseTree(int32_t node, std::unordered_set<int32_t>* unique_nodes,
                      std::vector<std::shared_ptr<Edge>>* edges) const;
};
