import os
import time
import datetime
import logging
import numpy as np
from sklearn import metrics as sklearn_metrics
import torch
from data_processor import seq_cls_load_and_cache_examples as load_data
from torch.utils.data import DataLoader, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
import onnxruntime as ort

logger = logging.getLogger(__name__)

def f1_pre_rec(labels, preds):
    return {
        "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
        "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
        "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        "acc": (labels == preds).mean()
    }

def evaluate(session, eval_dataset, device):
    eval_batch_size=64
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)


    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(eval_batch_size))

    preds = None
    out_label_ids = None

    for batch in progress_bar(eval_dataloader):

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": np.atleast_2d(batch[0].detach().cpu().numpy()),
                "attention_mask": np.atleast_2d(batch[1].detach().cpu().numpy()),
                "token_type_ids": np.atleast_2d(batch[2].detach().cpu().numpy()),
            }
            real_labels=batch[3]
            # print(inputs["input_ids"].shape)
            # print(inputs["attention_mask"].shape)
            # print(inputs["token_type_ids"].shape)
            outputs=session.run(None, inputs)
            logits = outputs[0]

            #print(logits)
            #print(real_labels)

        if preds is None:
            preds = logits
            out_label_ids = real_labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits, axis=0)
            out_label_ids = np.append(out_label_ids, real_labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    result = f1_pre_rec(out_label_ids, preds)
    results.update(result)

    print("***** Eval results *****")
    for key in results.keys():
        print("  {} = {}".format(key, str(results[key])))

    return results


output_dir = "./onnx_model_optimized"
export_model_path = os.path.join(output_dir, 'newnewtrain_kcelectra_fp16.onnx')
model_name_or_path="beomi/KcELECTRA-base"

# ONNX model inference
opt = ort.SessionOptions()
# opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# cuda_provider_options = {
# "arena_extend_strategy": "kSameAsRequested",
# }
# EP_list = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]
EP_list = ['CUDAExecutionProvider']
#EP_list = ['TensorrtExecutionProvider']
session = ort.InferenceSession(export_model_path, opt, providers=EP_list)

# load dataset
test_set=open('data/test_data.txt', 'r', encoding='utf-8').readlines()
test_dataset=load_data(test_set)
print(test_dataset)

# GPU or CPU
device = "cuda" if torch.cuda.is_available()  else "cpu"

start = time.time()
result = evaluate(session, test_dataset, device)
sec=time.time()-start
print(str(datetime.timedelta(seconds=sec)))