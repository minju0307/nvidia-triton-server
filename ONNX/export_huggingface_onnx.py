from typing import Mapping, OrderedDict
from transformers.onnx import OnnxConfig
from transformers import ElectraConfig, ElectraForSequenceClassification, AutoTokenizer
from pathlib import Path
from transformers.onnx import export
import os

model_name_or_path="beomi/KcELECTRA-base"
labels=["020121", "000001", "02051", "020811", "020819"]
max_seq_length=512

# custom ONNX configuration
class ElectraOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict([
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ])

model_dir = "newnew_train_kcelectra_128"

config = ElectraConfig.from_pretrained(
    model_dir,
    num_labels=len(labels),
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: i for i, label in enumerate(labels)},
)

#print(config)

onnx_config = ElectraOnnxConfig(config, task="sequence-classification") #seqcls onnx config

#print(onnx_config)
#print(onnx_config.default_onnx_opset)
#print(onnx_config.outputs)

# Exporting the model
output_dir = "./onnx_model_128"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

onnx_path=os.path.join(output_dir, 'newnewtrain_kcelectra_128.onnx')
onnx_path=Path(onnx_path)

model = ElectraForSequenceClassification.from_pretrained(model_dir, model_max_length=128)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)

onnx_inputs, onnx_outputs = \
    export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)


print('Model exported at', onnx_path)










