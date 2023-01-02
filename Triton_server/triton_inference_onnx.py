from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler, DataLoader
import torch
import tritonhttpclient
from transformers import AutoTokenizer
import numpy as np
import time
import datetime
from sklearn import metrics as sklearn_metrics
from fastprogress.fastprogress import master_bar, progress_bar


def simple_accuracy(labels, preds):
    return (labels == preds).mean()

# data processing
test_data=open('data/auc/test.tsv').readlines()
texts=[]
labels=[]
for line in test_data:
    try:
        text, label = line.split('\t')
        texts.append(text)
        labels.append(label)
    except:
        continue

tokenizer=AutoTokenizer.from_pretrained('beomi/KcELECTRA-base')

max_len=128
inputs=[]
for text in texts:
    inputs.append(tokenizer.batch_encode_plus(
        [text],
        max_length=max_len,
        padding="max_length",
        truncation=True))

all_input_ids=torch.tensor([i.input_ids for i in inputs], dtype=torch.long)
all_token_type_ids=torch.tensor([i.token_type_ids for i in inputs], dtype=torch.long)
all_attention_mask=torch.tensor([i.attention_mask for i in inputs], dtype=torch.long)

labels=[l.strip() for l in labels]
labels=[int(l) for l in labels]
all_labels=torch.tensor(labels)

dataset=TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=64)
results = {}

VERBOSE = False
input_name = ['input_ids', 'attention_mask', 'token_type_ids']
output_name = 'logits'
logits_all=[]
labels_all=[]

def run_inference(batch, model_name='kcelectra', url='127.0.0.1:8000', model_version='2'):

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_type_ids = batch["token_type_ids"]
    input_ids = np.array(input_ids, dtype=np.int64)
    attention_mask = np.array(attention_mask, dtype=np.int64)
    token_type_ids = np.array(token_type_ids, dtype=np.int64)

    num = len(input_ids)
    input_ids = input_ids.reshape(num, max_len)
    attention_mask = attention_mask.reshape(num, max_len)
    token_type_ids = token_type_ids.reshape(num, max_len)

    input0 = tritonhttpclient.InferInput(input_name[0], (num, max_len), 'INT64')
    input0.set_data_from_numpy(input_ids, binary_data=False)
    input1 = tritonhttpclient.InferInput(input_name[1], (num, max_len), 'INT64')
    input1.set_data_from_numpy(attention_mask, binary_data=False)
    input2 = tritonhttpclient.InferInput(input_name[2], (num, max_len), 'INT64')
    input2.set_data_from_numpy(token_type_ids, binary_data=False)

    output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=False)
    response = triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1, input2],
                                   outputs=[output])
    logits = response.as_numpy('logits')
    logits = np.array(logits, dtype=np.float32)

    logits_all.append(logits)

start=time.time()

model_name='kcelectra'
url='127.0.0.1:8000'
model_version='2'
triton_client = tritonhttpclient.InferenceServerClient(
    url=url, verbose=VERBOSE)
model_metadata = triton_client.get_model_metadata(
    model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(
    model_name=model_name, model_version=model_version)

print()
print("*** start inference ***")
print()
for batch in progress_bar(eval_dataloader):
    with torch.no_grad():
        inputs={"input_ids":batch[0], "attention_mask":batch[1], "token_type_ids":batch[2], "labels":batch[3]}
        run_inference(inputs)
        labels_all.append(inputs["labels"].detach().cpu().numpy())
sec=time.time()-start

print()
print()
print("*** inference time ***")
print(str(datetime.timedelta(seconds=sec)))

preds = []
for batch in logits_all:
    preds.extend(np.argmax(batch, axis=1))

real = []
for batch in labels_all:
    real.extend([i for i in batch])

result={
        "precision": sklearn_metrics.precision_score(real, preds, average="macro"),
        "recall": sklearn_metrics.recall_score(real, preds, average="macro"),
        "f1": sklearn_metrics.f1_score(real, preds, average="macro"),
        "f1_weighted": sklearn_metrics.f1_score(real, preds, average="weighted"),
        "acc": simple_accuracy(np.array(real), np.array(preds))
}

print()
print()
print("*** evaluation results ***")
print(result)