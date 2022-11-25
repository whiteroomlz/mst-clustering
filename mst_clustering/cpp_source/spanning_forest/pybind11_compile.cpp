#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "spanning_forest.h"

namespace py = pybind11;

void initSpanningForest(py::module &);

PYBIND11_MODULE(spanning_forest, m) {
    initSpanningForest(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
