// File: policy_convnet.cpp
// Python bindings for policy convnet model evaluator used in search inference

#include "python/model/policy_convnet.h"

#include <absl/strings/str_split.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "model/device_manager.h"
#include "model/model_evaluator.h"
#include "model/policy_convnet/policy_convnet_wrapper.h"

namespace py = pybind11;

namespace hpts::bindings {

using namespace model;
using namespace model::wrapper;

auto init_policy_convnet_levin(const PolicyConvNetConfig &config, double learning_rate, double weight_decay,
                               const std::string &devices, const std::string &output_path) {
    auto device_manager = std::make_unique<DeviceManager<PolicyConvNetWrapperLevin>>();
    for (const absl::string_view &device : absl::StrSplit(devices, ',')) {
        device_manager->AddDevice(
            std::make_unique<PolicyConvNetWrapperLevin>(config, learning_rate, weight_decay, std::string(device), output_path));
    }
    // Put this in return type
    return std::make_shared<ModelEvaluator<PolicyConvNetWrapperLevin>>(std::move(device_manager), 1);
}

void declare_model_evaluator_policy_convnet(py::module &m) {
    // Configuration
    py::class_<PolicyConvNetConfig>(m, "PolicyConvNetConfig")
        .def(py::init<const ObservationShape &, int, int, int, int, const std::vector<int> &, bool>())
        .def("__copy__", [](const PolicyConvNetConfig &self) { return PolicyConvNetConfig(self); })
        .def("__deepcopy__", [](const PolicyConvNetConfig &self, py::dict) { return PolicyConvNetConfig(self); })
        .def_readwrite("observation_shape", &PolicyConvNetConfig::observation_shape)
        .def_readwrite("num_actions", &PolicyConvNetConfig::num_actions)
        .def_readwrite("resnet_channels", &PolicyConvNetConfig::resnet_channels)
        .def_readwrite("resnet_blocks", &PolicyConvNetConfig::resnet_blocks)
        .def_readwrite("policy_channels", &PolicyConvNetConfig::policy_channels)
        .def_readwrite("policy_mlp_layers", &PolicyConvNetConfig::policy_mlp_layers)
        .def_readwrite("use_batchnorm", &PolicyConvNetConfig::use_batchnorm);

    // Inference Input
    using InferenceInput = PolicyConvNetWrapperLevin::InferenceInput;
    py::class_<InferenceInput>(m, "PolicyConvNetInferenceInput")
        .def(py::init<const Observation &>())
        .def("__copy__", [](const InferenceInput &self) { return InferenceInput(self); })
        .def("__deepcopy__", [](const InferenceInput &self, py::dict) { return InferenceInput(self); })
        .def_readwrite("observation", &InferenceInput::observation);

    // Inference Output
    using InferenceOutput = PolicyConvNetWrapperLevin::InferenceOutput;
    py::class_<InferenceOutput>(m, "PolicyConvNetInferenceOutput")
        .def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>())
        .def("__copy__", [](const InferenceOutput &self) { return InferenceOutput(self); })
        .def("__deepcopy__", [](const InferenceOutput &self, py::dict) { return InferenceOutput(self); })
        .def_readwrite("logits", &InferenceOutput::logits)
        .def_readwrite("policy", &InferenceOutput::policy)
        .def_readwrite("log_policy", &InferenceOutput::log_policy);

    // Learning Input
    using LearningInput = PolicyConvNetWrapperLevin::LearningInput;
    py::class_<LearningInput>(m, "PolicyConvNetLearningInput")
        .def(py::init<const Observation &, int, int>())
        .def("__copy__", [](const LearningInput &self) { return LearningInput(self); })
        .def("__deepcopy__", [](const LearningInput &self, py::dict) { return LearningInput(self); })
        .def_readwrite("observation", &LearningInput::observation)
        .def_readwrite("target_action", &LearningInput::target_action)
        .def_readwrite("expansions", &LearningInput::solution_expanded);

    // Evaluator
    using EvaluatorT = ModelEvaluator<PolicyConvNetWrapperLevin>;
    py::class_<EvaluatorT, std::shared_ptr<EvaluatorT>>(m, "PolicyConvNetEvaluator")
        .def(py::init(&init_policy_convnet_levin))
        .def("inference", &EvaluatorT::Inference)
        .def("learn",
             [](EvaluatorT &self, std::vector<LearningInput> &batch) {
                 auto device_manager = self.get_device_manager();
                 auto model = device_manager->Get(1, 0);
                 return model->Learn(batch);
             })
        .def("print", &EvaluatorT::print)
        .def("load", &EvaluatorT::load)
        .def("load_without_optimizer", &EvaluatorT::load_without_optimizer)
        .def("checkpoint_and_sync", &EvaluatorT::checkpoint_and_sync)
        .def("checkpoint_and_sync_without_optimizer", &EvaluatorT::checkpoint_and_sync_without_optimizer)
        .def("save_checkpoint", &EvaluatorT::save_checkpoint)
        .def("save_checkpoint_without_optimizer", &EvaluatorT::save_checkpoint_without_optimizer);
}

}    // namespace hpts::bindings
