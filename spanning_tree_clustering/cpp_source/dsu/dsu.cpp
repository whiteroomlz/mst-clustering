#include <iostream>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

class DSU {
public:
    DSU(size_t size);

    int32_t find(const int32_t& item);

    void unite(int32_t first, int32_t second);

    bool isSingleton() const;

private:
    std::vector<int32_t> data_;
    std::vector<size_t> weights_;
    bool is_singleton_ = false;
};

namespace py = pybind11;

PYBIND11_MODULE(dsu, m) {
    py::class_<DSU>(m, "DSU")
        .def(py::init<size_t>())
        .def("find", &DSU::find, "Returns the delegate of the unity the item belonging to.",
            py::arg("item"))
        .def("unite", &DSU::unite, "Unites the delegates of first and second items.",
            py::arg("first"), py::arg("second"))
        .def("is_singleton", &DSU::isSingleton, "Returns True if the DSU contains only one unity and False otherwise.");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

DSU::DSU(size_t size) : data_(std::vector<int32_t>(size)), weights_(std::vector<size_t>(size, 1)) {
    for (size_t index = 0; index < size; ++index) {
        data_[index] = index;
    }
}

int32_t DSU::find(const int32_t& item) {
    if (item == data_[item]) {
        return item;
    }

    return data_[item] = find(data_[item]);
}

void DSU::unite(int32_t first, int32_t second) {
    int32_t first_delegate = find(first);
    int32_t second_delegate = find(second);

    if (weights_[first_delegate] < weights_[second_delegate]) {
        std::swap(first_delegate, second_delegate);
    }

    data_[second_delegate] = first_delegate;
    weights_[first_delegate] += weights_[second_delegate];

    if (weights_[first_delegate] == data_.size()) {
        is_singleton_ = true;
    }
}

bool DSU::isSingleton() const {
    return is_singleton_;
}
