#!/usr/bin/env python

import collections

import safetensors
import torch
from torch import nn
from transformers import BertConfig, BertModel

all_weights = safetensors.torch.load_file("./output/best.safetensors")
print("Weights loaded", len(all_weights), "tensors")

bert_config = BertConfig.from_pretrained("./output/bert-config.json")
print("BERT config loaded")


# This model replicates the model described in the Rust part.
class MyModel(nn.Module):
    bert: BertModel
    classification: nn.Sequential

    def __init__(self, model: BertModel):
        super().__init__()
        self.bert = model
        self.classification = nn.Sequential(
            collections.OrderedDict(
                [("classifier1", nn.Linear(bert_config.hidden_size, 1))]
            )
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert.forward(input_ids, attention_mask)
        return self.classification.forward(output.last_hidden_state[:, 0, :])


bert_model = BertModel(bert_config, add_pooling_layer=False)
print("BERT loaded")

model = MyModel(bert_model)

model.load_state_dict(all_weights)
model.cuda()
model.eval()


def export(model, output):
    input_ids = torch.randint(1000, (3, 512), dtype=torch.int64).cuda()
    attention_mask = torch.randint(1, (3, 512), dtype=torch.int64).cuda()
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output,
        export_params=True,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Exported the model to {output}")


with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.float16):
        export(model, "output/best_fp16.onnx")
    export(model, "output/best.onnx")
