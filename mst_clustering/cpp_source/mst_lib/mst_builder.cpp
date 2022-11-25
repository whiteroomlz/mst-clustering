#include "mst_builder.h"

#include <algorithm>
#include <cfloat>
#include <functional>
#include <thread>
#include <utility>

void initMstBuilder(py::module& m) {
    py::class_<MstBuilder> mst_builder(m, "MstBuilder");
    mst_builder
        .def(py::init<const py::list&>(), py::arg("points"))
        .def("build", &MstBuilder::build, "Returns the minimal spanning tree",
                    py::arg("threads_count"), py::arg("distance_measure"));

    py::enum_<MstBuilder::Measure>(mst_builder, "DistanceMeasure")
        .value("COSINE", MstBuilder::Measure::COSINE)
        .value("EUCLIDEAN", MstBuilder::Measure::EUCLIDEAN)
        .value("QUADRATIC", MstBuilder::Measure::QUADRATIC)
        .export_values();
}

MstBuilder::MstBuilder(const py::list& points)
    : nodes_count_(points.size()),
      nearest_neighbours_(points.size()),
      points_(points.size()),
      used_(points.size()) {
    for (size_t point_index = 0; point_index < points.size(); ++point_index) {
        points_[point_index] = points[point_index].cast<std::vector<float>>();
    }
}

SpanningForest MstBuilder::build(size_t threads_count, Measure measure) {
    SpanningForest spanning_forest(points_.size());

    used_[0] = true;
    for (int32_t node = 1; node < nodes_count_; node++) {
        double distance;
        switch (measure) {
            case COSINE:
                distance = cosineDistance(points_[0], points_[node]);
                break;
            case EUCLIDEAN:
                distance = euclideanDistance(points_[0], points_[node]);
                break;
            case QUADRATIC:
                distance = quadraticDistance(points_[0], points_[node]);
                break;
        }
        nearest_neighbours_[node] = std::make_pair(distance, 0);
    }

    std::vector<std::thread> threads;
    threads.reserve(threads_count);
    std::vector<int32_t> threads_results(threads_count);
    size_t parallel_range_size = (nodes_count_ + threads_count - 1) / threads_count;

    for (int32_t iteration = 1; iteration < nodes_count_; iteration++) {
        for (int32_t thread_index = 0; thread_index < threads_count; thread_index++) {
            auto first = static_cast<int32_t>(thread_index * parallel_range_size);
            auto last = static_cast<int32_t>(std::min(nodes_count_, first + parallel_range_size));
            threads.emplace_back(&MstBuilder::findNearestNeighbourInRange, this, first, last,
                                 std::ref(threads_results[thread_index]));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();

        std::pair<double, int32_t> nearest_neighbour = std::make_pair(DBL_MAX, nodes_count_);
        for (int32_t node : threads_results) {
            if (node == nodes_count_) {
                continue;
            }
            nearest_neighbour =
                std::min({nearest_neighbours_[node].first, node}, nearest_neighbour);
        }

        int32_t node2 = nearest_neighbour.second;
        int32_t node1 = nearest_neighbours_[node2].second;
        used_[node2] = true;

        spanning_forest.addEdge(node1, node2, nearest_neighbour.first);

        for (int32_t thread_index = 0; thread_index < threads_count; thread_index++) {
            auto first = static_cast<int32_t>(thread_index * parallel_range_size);
            auto last = static_cast<int32_t>(std::min(nodes_count_, first + parallel_range_size));
            threads.emplace_back(&MstBuilder::updateDistances, this, first, last, node2, measure);
        }
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }

    return spanning_forest;
}

void MstBuilder::findNearestNeighbourInRange(int32_t first, int32_t last, int32_t& result) {
    std::pair<double, int32_t> nearest_neighbour = {DBL_MAX, nodes_count_};
    for (int32_t node = first; node < last; node++) {
        if (used_[node]) {
            continue;
        }
        nearest_neighbour = std::min(nearest_neighbour, {nearest_neighbours_[node].first, node});
    }
    result = nearest_neighbour.second;
}

void MstBuilder::updateDistances(int32_t first, int32_t last, int32_t point, Measure measure) {
    for (int32_t node = first; node < last; node++) {
        if (used_[node]) {
            continue;
        }
        double distance;
        switch (measure) {
            case COSINE:
                distance = cosineDistance(points_[point], points_[node]);
                break;
            case EUCLIDEAN:
                distance = euclideanDistance(points_[point], points_[node]);
                break;
            case QUADRATIC:
                distance = quadraticDistance(points_[point], points_[node]);
                break;
        }
        nearest_neighbours_[node] = std::min(nearest_neighbours_[node], {distance, point});
    }
}
