#include <unordered_map>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class Edge {
public:
    Edge(int32_t parent_id, int32_t child_id, double weight) : parent_id_(parent_id), child_id_(child_id), weight_(weight) {
    }

    int32_t getParentId() {
        return parent_id_;
    }

    int32_t getChildId() {
        return child_id_;
    }

    double getWeight() {
        return weight_;
    }

private:
    int32_t parent_id_;
    int32_t child_id_;
    double weight_;
};

class Node {
public:
    Node(int32_t id);

    Node(int32_t id, Node* parent);

    bool isRoot();

    int32_t id() const;

    Node* parent() const;

    void setParent(Node* parent);

    void addChild(int32_t id, double weight);

    void addChild(Node* node, double weight);

    Node* getNode(int32_t id);

    double getWeight(Node* child);

    static void erase(Node* child);

    void getAllEdges(py::list result);

    ~Node();

private:
    int32_t id_;
    Node* parent_;
    std::unordered_map<Node*, double> children_;
};

class SpanningForest {
public:
    size_t size() const;

    void addEdge(int32_t first_id, int32_t second_id, double weight);

    void removeEdge(int32_t first_id, int32_t second_id);

    void getRootEdges(int32_t index, py::list result) const;

    int32_t getRootId(int32_t index) const;

    ~SpanningForest();

private:
    std::vector<Node*> roots_;

    void reverse(Node* node);
};

PYBIND11_MODULE(spanning_forest, m) {
    py::class_<Edge>(m, "Edge")
        .def(py::init<int32_t, int32_t, double>(), py::arg("parent_id"), py::arg("child_id"), py::arg("weight"))
        .def("get_parent_id", &Edge::getParentId, "Returns the id of parent's node.")
        .def("get_child_id", &Edge::getChildId, "Returns the id of child's node.")
        .def("get_weight", &Edge::getWeight, "Returns the weight of the edge.");

    py::class_<SpanningForest>(m, "SpanningForest")
        .def(py::init<>())
        .def("size", &SpanningForest::size, "Returns the number of spanning trees in the forest.")
        .def("add_edge", &SpanningForest::addEdge, "Adds an edge between the two nodes with specified ids.",
            py::arg("first_id"), py::arg("second_id"), py::arg("weight"))
        .def("remove_edge", &SpanningForest::removeEdge, "Removes the edge between the two nodes with specified ids.",
            py::arg("first_id"), py::arg("second_id"))
        .def("get_root_edges", &SpanningForest::getRootEdges, "Returns all edges of root with the specified index.")
        .def("get_root_id", &SpanningForest::getRootId, "Returns id of root with the specified index.");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

Node::Node(int32_t id) : Node(id, nullptr) {
}

Node::Node(int32_t id, Node* parent) : id_(id), parent_(parent) {
}

bool Node::isRoot() {
    return parent_ == nullptr;
}

int32_t Node::id() const {
    return id_;
}

Node* Node::parent() const {
    return parent_;
}

void Node::setParent(Node* parent) {
    parent_ = parent;
}

void Node::addChild(int32_t id, double weight) {
    children_[new Node(id, this)] = weight;
}

void Node::addChild(Node* node, double weight) {
    node->parent_ = this;
    children_[node] = weight;
}

Node* Node::getNode(int32_t id) {
    if (id_ == id) {
        return this;
    }

    for (auto key_value_pair : children_) {
        Node* child = key_value_pair.first;

        Node* result = child->getNode(id);

        if (result != nullptr) {
            return result;
        }
    }

    return nullptr;
}

double Node::getWeight(Node* child) {
    return children_[child];
}

void Node::erase(Node* child) {
    child->parent_->children_.erase(child);
}

void Node::getAllEdges(py::list result) {
    if (!children_.empty()) {
        for (auto key_value_pair : children_) {
            Node* child = key_value_pair.first;

            result.append(Edge(id_, child->id_, getWeight(child)));
            child->getAllEdges(result);
        }
    }
}

Node::~Node() {
    for (auto key_value_pair : children_) {
        delete key_value_pair.first;
    }
}

size_t SpanningForest::size() const {
    return roots_.size();
}

void SpanningForest::addEdge(int32_t first_id, int32_t second_id, double weight) {
    Node* first_node = nullptr;
    Node* second_node = nullptr;
    Node* temp = nullptr;

    for (auto root : roots_) {
        if ((first_node = root->getNode(first_id)) != nullptr) {
            temp = root;
            break;
        }
    }

    for (auto root : roots_) {
        if ((second_node = root->getNode(second_id)) != nullptr) {
            if (root == temp) {
                std::stringstream message;
                message << "In result of adding edge [" << first_id << "; " << second_id << "] the cycle will be generated.";
                throw std::invalid_argument(message.str());
            }
            temp = root;
            break;
        }
    }

    if (first_node != nullptr && second_node != nullptr) {
        reverse(second_node);

        first_node->addChild(second_node, weight);
        second_node->setParent(first_node);

        for (auto iterator = roots_.begin(); iterator < roots_.end(); iterator++) {
            if (*iterator == temp) {
                roots_.erase(iterator);
                break;
            }
        }
    } else if (first_node != nullptr) {
        first_node->addChild(second_id, weight);

    } else if (second_node != nullptr) {
        second_node->addChild(first_id, weight);

    } else {
        Node* root = new Node(first_id);
        root->addChild(second_id, weight);
        roots_.emplace_back(root);
    }
}

void SpanningForest::removeEdge(int32_t first_id, int32_t second_id) {
    Node* first_node = nullptr;
    Node* second_node = nullptr;

    for (auto root : roots_) {
        if ((first_node = root->getNode(first_id)) != nullptr) {
            if ((second_node = root->getNode(second_id)) != nullptr) {
                if (second_node->parent() == first_node) {
                    first_node->erase(second_node);
                    second_node->setParent(nullptr);
                    roots_.emplace_back(second_node);
                    return;

                } else if (first_node->parent() == second_node) {
                    first_node->setParent(nullptr);
                    second_node->erase(first_node);
                    roots_.emplace_back(first_node);
                    return;

                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    std::stringstream message;
    message << "The edge [" << first_id << "; " << second_id << "] does not exist.";
    throw std::invalid_argument(message.str());
}

void SpanningForest::getRootEdges(int32_t index, py::list result) const {
    roots_.at(index)->getAllEdges(result);
}

int32_t SpanningForest::getRootId(int32_t index) const {
    return roots_.at(index)->id();
}

void SpanningForest::reverse(Node* node) {
    if (node->parent() == nullptr) {
        return;
    }

    reverse(node->parent());

    Node* parent = node->parent();
    node->addChild(parent, parent->getWeight(node));
    parent->erase(node);
}

SpanningForest::~SpanningForest() {
    for (auto root : roots_) {
        delete root;
    }
}
