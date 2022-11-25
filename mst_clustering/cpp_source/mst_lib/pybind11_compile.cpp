#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void initSpanningForest(py::module &);
void initMstBuilder(py::module &);

PYBIND11_MODULE(mst_lib, m) {
    initSpanningForest(m);
    initMstBuilder(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
