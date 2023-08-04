import os

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
import torch
import torch.nn as nn
import time
from logging import getLogger
logger = getLogger(__name__)

# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer, LlamaTokenizer
    if "llama" in model:
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset, testenc

@torch.no_grad()
def llama_eval(model, testenc, dev, seqlen = 2048):
    from tqdm import tqdm
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        print('layer', i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm(range(nsamples)):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print('ppl: ')
    print(ppl.item())
    print()

    model.config.use_cache = use_cache

@torch.no_grad()
def opt_eval(model, testenc, dev, seqlen = 2048):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def eval(model_base, model_checkpoint, eval_target):
    from datasets import load_dataset
    from datautils import get_loaders, Evaluator_lambada, Evaluator_piqa, Evaluator_hellaswag
    time_start = time.time()

    # datasets = ['wikitext2', 'ptb', 'c4-new']
    datasets = []
    if 'wikitext2' in eval_target: datasets.append('wikitext2')
    if 'ptb' in eval_target: datasets.append('ptb')
    if 'c4-new' in eval_target: datasets.append('c4-new')
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=model_base, seqlen=2048
        )
        print(dataset)
        llama_eval(model_checkpoint, testloader, 'cuda')

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_base, use_fast=False)
    tokenizer.pad_token = "[PAD]"

    # ============lambada==================
    if 'lambada' in eval_target:
        dataset = load_dataset('lambada', split='validation', cache_dir='/gptq/datasets')
        evaluator_lambada = Evaluator_lambada(dataset, tokenizer, 'cuda')
        acc_lambada = evaluator_lambada.evaluate(model_checkpoint.cuda())
        evaluator_lambada = None
        dataset = None
        print("lambada: ", acc_lambada)
        torch.cuda.empty_cache()
    # =============piqa=====================
    if 'piqa' in eval_target:
        dataset = load_dataset('piqa', split='validation', cache_dir='/gptq/datasets')
        evaluator_piqa = Evaluator_piqa(dataset, tokenizer, 'cuda', model_checkpoint)
        acc_piqa = evaluator_piqa.evaluate(model_checkpoint.cuda())
        evaluator_piqa = None
        dataset = None
        print("piqa: ", acc_piqa)
        torch.cuda.empty_cache()
    # =============hellaswag================
    if 'hellaswag' in eval_target:
        dataset = load_dataset('hellaswag', split='validation', cache_dir='/gptq/datasets')
        evaluator_hellaswag = Evaluator_hellaswag(dataset, tokenizer, 'cuda', model_checkpoint)
        acc_hellaswag = evaluator_hellaswag.evaluate(model_checkpoint.cuda())
        evaluator_hellaswag = None
        dataset = None
        print("hellaswag: ", acc_hellaswag)
        torch.cuda.empty_cache()

    print('eval time: %fh' % ((time.time() - time_start) / 60. / 60.))

def mmlu_eval(model, output_path=None):
    import json
    from lm_eval import tasks, evaluator, utils
    tasks_ = 'hendrycksTest-*'
    task_names = utils.pattern_match(tasks_.split(","), tasks.ALL_TASKS)
    # task_names = ['hendrycksTest-world_religions']
    print(f"Selected Tasks: {task_names}")

    time_start = time.time()
    results = evaluator.simple_evaluate(
        model=model.to('cuda:0'),
        model_args='use_accelerate=True',
        tasks=task_names,
        num_fewshot=5,
        batch_size=2,
    )
    print('mmlu eval time: %fh' % ((time.time() - time_start) / 60. / 60.))

    dumped = json.dumps(results, indent=2)
    # print(dumped)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(dumped)

    print(evaluator.make_table(results))

    # 初始化总 acc 和计数器
    acc_sum = 0
    count = 0

    # 遍历所有 hendrycksTest 相关的数据
    for key in results['results']:
        if 'hendrycksTest' in key:
            # 累加 acc 值并增加计数器
            acc_sum += results['results'][key]['acc']
            count += 1

    print("Num of tests", count)
    # 计算平均值
    avg_acc = acc_sum / count

    print("mmlu-acc:", avg_acc)

# pretrained_model_dir = "huggyllama/llama-7b"
# quantized_model_dir = "save/llama1-7b_nf4_g128"

# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "save/opt125m_int4_test"

pretrained_model_dir = "meta-llama/Llama-2-7b-hf"
# quantized_model_dir = "save/llama2-7b_nf4_g64"

# pretrained_model_dir = "meta-llama/Llama-2-13b-hf"
# quantized_model_dir = "save/llama2-13b_nf4_g128"

# pretrained_model_dir = "meta-llama/Llama-2-70b-hf"
# quantized_model_dir = "save/llama2-70b_nf4_g128"

# pretrained_model_dir = "meta-llama/Llama-2-70b-chat-hf"
# quantized_model_dir = "save/llama2-70b_chat_nf4_g128"


def main():
    do_quant = True # quant or only load&eval ori fp16 model
    format = 'fp'
    group_size = 128
    pack = True # If only quant & eval in fp16, no pack. If need to save 4bit model, pack

    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    traindataset,testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        format=format, # quantize model to int / nf / fp
        group_size=group_size,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, cache_dir='/gptq_hub')

    # quantize model
    if do_quant:
        time_start = time.time()
        model.quantize(traindataset, use_triton=False, pack=pack)
        logger.info('quant time: %fh' % ((time.time() - time_start) / 60. / 60.))

    # model.save_quantized(quantized_model_dir)
    # model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, 
    #     inject_fused_attention=False, inject_fused_mlp=False
    # )

    # wikitext eval for llama/opt, from ori augogptq code
    # if 'llama' in pretrained_model_dir:
    #     llama_eval(model.model, testenc, "cuda:0")
    # else:
    #     opt_eval(model.model, testenc, "cuda:0")

    # ['wikitext2', 'ptb', 'c4-new', 'lambada', 'piqa', 'hellaswag'] eval
    # if '7b' in pretrained_model_dir:
    #     model_base = 'meta-llama/Llama-2-7b-hf'
    # elif '13b' in pretrained_model_dir:
    #     model_base = 'meta-llama/Llama-2-13b-hf'
    # elif '70b' in pretrained_model_dir:
    #     model_base = 'meta-llama/Llama-2-70b-hf'
    # # datasets = ['wikitext2', 'ptb', 'c4-new', 'lambada', 'piqa', 'hellaswag']
    # datasets = ['wikitext2']
    # eval(model_base, model.model, datasets)

    # mmlu eval
    mmlu_eval(model.model)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
