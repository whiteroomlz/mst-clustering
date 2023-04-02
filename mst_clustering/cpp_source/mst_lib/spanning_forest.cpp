#include "spanning_forest.h"

PYBIND11_MAKE_OPAQUE(std::vector<int32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<SpanningForest::Edge>>);

void initSpanningForest(py::module& m) {
    py::class_<SpanningForest::Edge, std::shared_ptr<SpanningForest::Edge>>(m, "Edge")
        .def(py::init<int32_t, int32_t, double>(), py::arg("parent_id"), py::arg("child_id"),
             py::arg("weight"))
        .def_readonly("first_node", &SpanningForest::Edge::first_node)
        .def_readonly("second_node", &SpanningForest::Edge::second_node)
        .def_readonly("weight", &SpanningForest::Edge::edge_weight)
        .def("nodes", &SpanningForest::Edge::nodes)
        .def("__repr__", [](const SpanningForest::Edge& edge) {
            return (std::stringstream("")
                    << "SpanningForest::Edge - (" << edge.first_node << ";" << edge.second_node
                    << ") with weight " << edge.edge_weight)
                .str();
        });

    py::class_<SpanningForest>(m, "SpanningForest")
        .def(py::init<size_t>(), py::arg("capacity"))
        .def("size", &SpanningForest::size, "Returns the number of spanning trees in the forest.")
        .def("is_spanning_tree", &SpanningForest::isSpanningTree,
             "Returns true if the forest contains only one root.")
        .def("find_root", &SpanningForest::findRoot, "Returns the root of the specified node.",
             py::arg("node"))
        .def("get_roots", &SpanningForest::getRoots, "Returns all roots ids.")
        .def("get_tree_info", &SpanningForest::getTreeInfo,
             "Returns all nodes and edges of root with the specified id.", py::arg("root"),
             py::arg("out_edges"))
        .def("get_tree_nodes", &SpanningForest::getTreeNodes,
             "Returns all nodes and edges of root with the specified id.", py::arg("root"))
        .def("add_edge", &SpanningForest::addEdge,
             "Adds an edge between the two nodes with specified ids.", py::arg("first_node"),
             py::arg("second_node"), py::arg("edge_weight"))
        .def("remove_edge", &SpanningForest::removeEdge,
             "Removes the edge between the two nodes with specified ids.", py::arg("first_node"),
             py::arg("second_node"));

    py::bind_vector<std::vector<int32_t>>(m, "Int32Vector");
    py::bind_vector<std::vector<std::shared_ptr<SpanningForest::Edge>>>(m, "EdgeVector");
}

// O(1).
size_t SpanningForest::size() const {
    return trees_count_;
}

// O(1).
bool SpanningForest::isSpanningTree() const {
    return trees_count_ == 1;
}

// O(a(N)), где N - число точек.
int32_t SpanningForest::findRoot(int32_t node) {
    if (node == roots_[node]) {
        return node;
    }
    return roots_[node] = findRoot(roots_[node]);
}

// O(r), где r - число деревьев в лесу.
void SpanningForest::getRoots(std::vector<int32_t>& out_roots) const {
    out_roots = {unique_roots_.begin(), unique_roots_.end()};
}

// O(m), где m - число вершин в выбранном остовном дереве.
py::array_t<int32_t> SpanningForest::getTreeInfo(int32_t root,
                                                 std::vector<std::shared_ptr<Edge>>& out_edges) {
    std::unordered_set<int32_t> unique_nodes;
    std::vector<std::shared_ptr<Edge>> edges;
    traverseTree(root, &unique_nodes, &edges);

    out_edges = std::move(edges);

    auto nodes = py::array_t<int32_t>(static_cast<int64_t>(unique_nodes.size()));
    py::buffer_info buffer_info = nodes.request();
    auto* buffer = static_cast<int32_t*>(buffer_info.ptr);
    auto node_ptr = unique_nodes.begin();
    for (int node_index = 0; node_index < nodes.size(); ++node_index, ++node_ptr) {
        buffer[node_index] = *node_ptr;
    }

    return nodes;
}

py::array_t<int32_t> SpanningForest::getTreeNodes(int32_t root) {
    std::unordered_set<int32_t> unique_nodes;
    traverseTree(root, &unique_nodes, nullptr);

    auto nodes = py::array_t<int32_t>(static_cast<int64_t>(unique_nodes.size()));
    py::buffer_info buffer_info = nodes.request();
    auto* buffer = static_cast<int32_t*>(buffer_info.ptr);
    auto node_ptr = unique_nodes.begin();
    for (int node_index = 0; node_index < nodes.size(); ++node_index, ++node_ptr) {
        buffer[node_index] = *node_ptr;
    }

    return nodes;
}

// O(a(N)), где N - число точек.
void SpanningForest::addEdge(int32_t first_node, int32_t second_node, double edge_weight) {
    dsuUnite(first_node, second_node);

    std::shared_ptr<Edge> edge(new Edge{first_node, second_node, edge_weight});
    edges_.emplace(first_node, edge);
    edges_.emplace(second_node, edge);

    --trees_count_;
}

// O(m * a(N)), где m - число вершин в меньшем остовном дереве (после разбиения), а N - число точек.
void SpanningForest::removeEdge(int32_t first_node, int32_t second_node) {
    auto first_values_range = edges_.equal_range(first_node);
    for (auto key_value = first_values_range.first; key_value != first_values_range.second;
         ++key_value) {
        if ((*key_value).second->second_node == second_node) {
            edges_.erase(key_value);
            break;
        }
    }

    auto second_values_range = edges_.equal_range(second_node);
    for (auto key_value = second_values_range.first; key_value != second_values_range.second;
         ++key_value) {
        if ((*key_value).second->first_node == first_node) {
            edges_.erase(key_value);
            break;
        }
    }

    std::unordered_set<int32_t> second_tree_nodes;
    std::vector<std::shared_ptr<Edge>> edges;

    traverseTree(second_node, &second_tree_nodes, &edges);
    if (second_tree_nodes.find(findRoot(second_node)) != second_tree_nodes.end()) {
        std::swap(first_node, second_node);
        second_tree_nodes.clear();
        edges.clear();
        traverseTree(second_node, &second_tree_nodes, &edges);
    }

    dsu_weights_[findRoot(first_node)] -= second_tree_nodes.size();

    for (const int32_t node : second_tree_nodes) {
        unique_roots_.insert(node);
        roots_[node] = node;
        dsu_weights_[node] = 1;
    }

    for (const std::shared_ptr<Edge>& edge : edges) {
        dsuUnite(edge->first_node, edge->second_node);
    }

    ++trees_count_;
}

// O(a(N)), где N - число точек.
void SpanningForest::dsuUnite(int32_t first_node, int32_t second_node) {
    int32_t first_parent = findRoot(first_node);
    int32_t second_parent = findRoot(second_node);

    if (first_parent == second_parent) {
        std::stringstream message;
        message << "Uniting the passed nodes {" << first_node << "} and {" << second_node
                << "} will produce a cycle.";
        throw std::invalid_argument(message.str());
    }

    if (dsu_weights_[first_parent] < dsu_weights_[second_parent]) {
        std::swap(first_parent, second_parent);
    }

    unique_roots_.erase(second_parent);
    roots_[second_parent] = first_parent;
    dsu_weights_[first_parent] += dsu_weights_[second_parent];
}

// O(2 * (m - 1)), где m - число нод в кластере, т. к. для каждой функция будет вызвана 1 раз.
void SpanningForest::traverseTree(int32_t node, std::unordered_set<int32_t>* unique_nodes,
                                  std::vector<std::shared_ptr<Edge>>* edges) const {
    unique_nodes->insert(node);

    auto range = edges_.equal_range(node);
    // O(l), где l - число рёбер, включающих вершину node.
    // Всего явно хранится в одном остовном дереве 2 * (m - 1) рёбер.
    for (auto key_value = range.first; key_value != range.second; ++key_value) {
        std::shared_ptr<Edge> edge = (*key_value).second;
        int32_t other_node = edge->first_node == node ? edge->second_node : edge->first_node;
        if (unique_nodes->find(other_node) == unique_nodes->end()) {
            if (edges) {
                edges->emplace_back(edge);
            }

            traverseTree(other_node, unique_nodes, edges);
        }
    }
}
