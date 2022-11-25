#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <vector>

#include "point.h"
#include "spanning_forest.h"

namespace py = pybind11;

class MstBuilder {
public:
    enum Measure { COSINE = 0, EUCLIDEAN, QUADRATIC };

    explicit MstBuilder(const py::list& points);

    SpanningForest build(size_t threads_count, Measure measure);

private:
    void findNearestNeighbourInRange(int32_t first, int32_t last, int32_t& result);

    void updateDistances(int32_t first, int32_t last, int32_t point, Measure measure);

    size_t nodes_count_;
    std::vector<Point> points_;
    std::vector<std::pair<double, int32_t>> nearest_neighbours_;
    std::vector<bool> used_;
};
