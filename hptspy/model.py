import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

from _hptspy import _model  # type: ignore

if TYPE_CHECKING:
    from .common import ObservationShape

# Bring names into current scope

# Policy ConvNet
PolicyConvNetConfig = _model.PolicyConvNetConfig
PolicyConvNetEvaluator = _model.PolicyConvNetEvaluator
PolicyConvNetInferenceInput = _model.PolicyConvNetInferenceInput
PolicyConvNetInferenceOutput = _model.PolicyConvNetInferenceOutput
PolicyConvNetLearningInput = _model.PolicyConvNetLearningInput


# TwoHeaded ConvNet
TwoHeadedConvNetConfig = _model.TwoHeadedConvNetConfig
TwoHeadedConvNetEvaluator = _model.TwoHeadedConvNetEvaluator
TwoHeadedConvNetInferenceInput = _model.TwoHeadedConvNetInferenceInput
TwoHeadedConvNetInferenceOutput = _model.TwoHeadedConvNetInferenceOutput
TwoHeadedConvNetLearningInput = _model.TwoHeadedConvNetLearningInput


class _ModelBase:
    def __init__(self, model_eval):
        self.model_eval = model_eval

    def print(self):
        self.model_eval.print()

    def load(self, step: int):
        self.model_eval.load(step)

    def load_without_optimizer(self, step: int):
        self.model_eval.load_without_optimizer(step)

    def checkpoint_and_sync(self, step: int):
        self.model_eval.checkpoint_and_sync(step)

    def checkpoint_and_sync_without_optimizer(self, step: int):
        self.model_eval.checkpoint_and_sync_without_optimizer(step)

    def save_checkpoint(self, step: int):
        self.model_eval.save_checkpoint(step)

    def save_checkpoint_without_optimizer(self, step: int):
        self.model_eval.save_checkpoint_without_optimizer(step)

    def inference(self, _):
        raise NotImplementedError()

    def learn(self, _):
        raise NotImplementedError()


class PolicyConvNet(_ModelBase):
    ModelType = "policy_convnet"
    Config = PolicyConvNetConfig
    Evaluator = PolicyConvNetEvaluator
    InferenceInput = PolicyConvNetInferenceInput
    InferenceOutput = PolicyConvNetInferenceOutput
    LearningInput = PolicyConvNetLearningInput

    def __init__(
        self,
        model_config_json: str,
        observation_shape: ObservationShape,
        num_actions: int,
        output_path: str,
    ):
        model_config = json.loads(model_config_json)
        model_type = PolicyConvNet.ModelType
        if model_config["model_type"] != model_type:
            raise Exception("field 'model_type' must be '{}'".format(model_type))
        config = PolicyConvNetConfig(
            observation_shape,
            num_actions,
            model_config["resnet_channels"],
            model_config["resnet_blocks"],
            model_config["policy_channels"],
            model_config["policy_mlp_layers"],
            model_config["use_batchnorm"],
        )
        model_eval = PolicyConvNetEvaluator(
            config,
            model_config["learning_rate"],
            model_config["weight_decay"],
            model_config["devices"],
            output_path,
        )
        super().__init__(model_eval)

    def inference(self, batch: list[InferenceInput]) -> InferenceOutput:
        return self.model_eval.inference(batch)

    def learn(self, batch: list[LearningInput]) -> float:
        return self.model_eval.learn(batch)


class TwoHeadedConvNet(_ModelBase):
    ModelType = "twoheaded_convnet"
    Config = TwoHeadedConvNetConfig
    Evaluator = TwoHeadedConvNetEvaluator
    InferenceInput = TwoHeadedConvNetInferenceInput
    InferenceOutput = TwoHeadedConvNetInferenceOutput
    LearningInput = TwoHeadedConvNetLearningInput

    def __init__(
        self,
        model_config_json: str,
        observation_shape: ObservationShape,
        num_actions: int,
        output_path: str,
    ):
        model_config = json.loads(model_config_json)
        model_type = TwoHeadedConvNet.ModelType
        if model_config["model_type"] != model_type:
            raise Exception("field 'model_type' must be '{}'".format(model_type))
        config = TwoHeadedConvNetConfig(
            observation_shape,
            num_actions,
            model_config["resnet_channels"],
            model_config["resnet_blocks"],
            model_config["policy_channels"],
            model_config["heuristic_channels"],
            model_config["policy_mlp_layers"],
            model_config["heuristic_mlp_layers"],
            model_config["use_batchnorm"],
        )
        model_eval = TwoHeadedConvNetEvaluator(
            config,
            model_config["learning_rate"],
            model_config["weight_decay"],
            model_config["devices"],
            output_path,
        )
        super().__init__(model_eval)

    def inference(self, batch: list[InferenceInput]) -> InferenceOutput:
        return self.model_eval.inference(batch)

    def learn(self, batch: list[LearningInput]) -> float:
        return self.model_eval.learn(batch)


@dataclass
class ModelProperties:
    Config: Type[PolicyConvNetConfig] | Type[TwoHeadedConvNetConfig]
    Evaluator: Type[PolicyConvNetEvaluator] | Type[TwoHeadedConvNetEvaluator]
    InferenceInput: Type[PolicyConvNetInferenceInput] | Type[
        TwoHeadedConvNetInferenceInput
    ]
    InferenceOutput: Type[PolicyConvNetInferenceOutput] | Type[
        TwoHeadedConvNetInferenceOutput
    ]
    LearningInput: Type[PolicyConvNetLearningInput] | Type[
        TwoHeadedConvNetLearningInput
    ]


# Wrapped type traits for supported models
# PolicyConvNet = ModelProperties(
#     PolicyConvNetConfig,
#     PolicyConvNetEvaluator,
#     PolicyConvNetInferenceInput,
#     PolicyConvNetInferenceOutput,
#     PolicyConvNetLearningInput,
# )
# TwoHeadedConvNet = ModelProperties(
#     TwoHeadedConvNetConfig,
#     TwoHeadedConvNetEvaluator,
#     TwoHeadedConvNetInferenceInput,
#     TwoHeadedConvNetInferenceOutput,
#     TwoHeadedConvNetLearningInput,
# )
