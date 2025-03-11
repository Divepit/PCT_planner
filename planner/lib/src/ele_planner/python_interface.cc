#include "ele_planner/offline_ele_planner.h"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(ele_planner, m)
{
     auto pyOfflineElePlanner = py::class_<OfflineElePlanner>(m, "OfflineElePlanner");
     pyOfflineElePlanner
         .def(py::init<double, bool>(), py::arg("max_heading_rate"), py::arg("use_quintic") = false)
         .def("init_map", [](OfflineElePlanner &self,
                             double a_start_cost_threshold,
                             double safe_cost_margin,
                             double resolution,
                             int num_layers,
                             double step_cost_weight,
                             py::array_t<double, py::array::c_style | py::array::forcecast> cost_map,
                             py::array_t<double, py::array::c_style | py::array::forcecast> height_map,
                             py::array_t<double, py::array::c_style | py::array::forcecast> ceiling,
                             py::array_t<double, py::array::c_style | py::array::forcecast> ele_map,
                             py::array_t<double, py::array::c_style | py::array::forcecast> grad_x,
                             py::array_t<double, py::array::c_style | py::array::forcecast> grad_y)
              {
            // Get buffer info for each array
            auto buf_cost = cost_map.request();
            auto buf_height = height_map.request();
            auto buf_ceiling = ceiling.request();
            auto buf_ele = ele_map.request();
            auto buf_grad_x = grad_x.request();
            auto buf_grad_y = grad_y.request();
            
            // Create Eigen matrices from numpy arrays (copies, not views)
            Eigen::MatrixXd cost_mat = Eigen::Map<Eigen::MatrixXd>(
                static_cast<double*>(buf_cost.ptr),
                buf_cost.shape[0], buf_cost.shape[1]);
                
            Eigen::MatrixXd height_mat = Eigen::Map<Eigen::MatrixXd>(
                static_cast<double*>(buf_height.ptr),
                buf_height.shape[0], buf_height.shape[1]);
                
            Eigen::MatrixXd ceiling_mat = Eigen::Map<Eigen::MatrixXd>(
                static_cast<double*>(buf_ceiling.ptr),
                buf_ceiling.shape[0], buf_ceiling.shape[1]);
                
            Eigen::MatrixXd ele_mat = Eigen::Map<Eigen::MatrixXd>(
                static_cast<double*>(buf_ele.ptr),
                buf_ele.shape[0], buf_ele.shape[1]);
                
            Eigen::MatrixXd grad_x_mat = Eigen::Map<Eigen::MatrixXd>(
                static_cast<double*>(buf_grad_x.ptr),
                buf_grad_x.shape[0], buf_grad_x.shape[1]);
                
            Eigen::MatrixXd grad_y_mat = Eigen::Map<Eigen::MatrixXd>(
                static_cast<double*>(buf_grad_y.ptr),
                buf_grad_y.shape[0], buf_grad_y.shape[1]);
            
            // Call the C++ method with our copies
            self.InitMap(a_start_cost_threshold, safe_cost_margin, resolution,
                        num_layers, step_cost_weight, cost_mat, height_mat,
                        ceiling_mat, ele_mat, grad_x_mat, grad_y_mat); })
         .def("plan", [](OfflineElePlanner &self,
                         py::array_t<int, py::array::c_style | py::array::forcecast> start,
                         py::array_t<int, py::array::c_style | py::array::forcecast> goal,
                         bool optimize)
              {
            // Get buffer info
            auto buf_start = start.request();
            auto buf_goal = goal.request();
            
            // Check dimensions
            if (buf_start.ndim != 1 || buf_start.shape[0] != 3 ||
                buf_goal.ndim != 1 || buf_goal.shape[0] != 3) {
                throw std::runtime_error("start and goal must be 3-element vectors");
            }
            
            // Create Eigen vectors (copies, not views)
            Eigen::Vector3i start_vec = Eigen::Map<Eigen::Vector3i>(
                static_cast<int*>(buf_start.ptr), 3);
                
            Eigen::Vector3i goal_vec = Eigen::Map<Eigen::Vector3i>(
                static_cast<int*>(buf_goal.ptr), 3);
                
            // Call the method with our copies
            return self.Plan(start_vec, goal_vec, optimize); })
         .def("debug", &OfflineElePlanner::Debug)
         .def("set_reference_height", &OfflineElePlanner::SetReferenceHeight)
         .def("set_max_iterations", &OfflineElePlanner::set_max_iterations)
         .def("get_path_finder", &OfflineElePlanner::get_path_finder)
         .def("get_map", &OfflineElePlanner::get_map)
         .def("get_trajectory_optimizer", &OfflineElePlanner::get_trajectory_optimizer)
         .def("get_trajectory_optimizer_wnoj", &OfflineElePlanner::get_trajectory_optimizer_wnoj)
         .def("get_debug_path", &OfflineElePlanner::GetDebugPath);
}