import argparse

import os
import sys
import csv
import torch
import numpy as np
import json
import multiprocessing as mp
import numpy as np
import time
import random

from functools import partial
from transformers import LlamaForCausalLM, LlamaTokenizer

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


import google.protobuf.json_format
from collections.abc import Mapping
from tritonclient.grpc.service_pb2 import ModelInferResponse


PWD = os.path.abspath(os.path.dirname(__file__))


def _token2inputs(FLAGS, token):
    input_ids = token.input_ids.numpy().astype(np.uint32)
    mem_seq_len = torch.sum(token.attention_mask, dim=1).numpy().astype(np.uint32)
    mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0],1])
    request_output_len = FLAGS.maximum_output_length * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
    runtime_top_k = (FLAGS.sampling_topk * np.ones([input_ids.shape[0],1])).astype(np.uint32)
    runtime_top_p = FLAGS.sampling_topp * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    temperature = 0.5 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.2 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    random_seed = random.randint(0, 1000) * np.ones([input_ids.shape[0], 1]).astype(np.uint64)
    is_return_log_probs = True * np.ones([input_ids.shape[0], 1]).astype(bool)
    bad_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
    stop_words_ids = np.array([[[2], [2]]] * input_ids.shape[0], dtype=np.int32)
    beam_width = (FLAGS.beam_width * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
    start_ids = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
    end_ids = 2 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)

    def to_input(name, np_input):
        protocol = "grpc"
        client_util = httpclient if protocol == "http" else grpcclient
        t = client_util.InferInput(
            name, np_input.shape, np_to_triton_dtype(np_input.dtype))
        t.set_data_from_numpy(np_input)
        return t        

    inputs = [to_input("input_ids", input_ids),
              to_input("input_lengths", mem_seq_len),
              to_input("request_output_len", request_output_len),
              to_input("runtime_top_k", runtime_top_k),
              to_input("runtime_top_p", runtime_top_p),
              to_input("beam_search_diversity_rate", beam_search_diversity_rate),
              to_input("temperature", temperature),
              to_input("len_penalty", len_penalty),
              to_input("repetition_penalty", repetition_penalty),
              to_input("random_seed", random_seed),
              to_input("is_return_log_probs", is_return_log_probs),
              to_input("beam_width", beam_width),
              to_input("start_id", start_ids),
              to_input("end_id", end_ids),
              to_input("bad_words_list", bad_words_ids),
              to_input("stop_words_list", stop_words_ids)]
    return inputs


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)
      
def stream_consumer(queue, test: bool):
    start_time = None
    tokens_before = np.array([], dtype=np.int32)
    while True:
        result = queue.get()
        if result is None:
            break

        if isinstance(result, float):
            start_time = result
            continue

        message = ModelInferResponse()
        google.protobuf.json_format.Parse(json.dumps(result), message)
        result = grpcclient.InferResult(message)

        idx = result.as_numpy("sequence_length")[0, 0]
        tokens = result.as_numpy("output_ids")[0, 0, :idx]
        prob = result.as_numpy("cum_log_probs")[0, 0]
        print("[After {:.2f}s] Partial result (probability {:.2e}):\n{}\n".format(
            time.perf_counter() - start_time, np.exp(prob), tokens))
        
        if test:
            assert len(tokens) == len(tokens_before) + 1
            assert np.array_equal(tokens[:-1], tokens_before)
            tokens_before = tokens

  
def token_printer(tokenizer, result, error):
    if error:
        print("[E:%s]".format(str(result)), end="")      # without "\n"
    else:
        result = grpcclient.InferResult(result.get_response())
        seq_len = result.as_numpy("sequence_length")[0, 0]
        token = np.array([result.as_numpy("output_ids")[0, 0, seq_len-1]])
        token_text = tokenizer.decode(token)
    sys.stdout.flush()
    
def stream_callback(queue, tokenizer, result, error):
    if error:
        print("error..........:", error)
        queue.put(error)
    else:
        queue.put(result.get_response(as_json=True))
        print("queue.put(result.get_response(as_json=True)):{}".format(result.get_response(as_json=True)))
        
        infer_result = grpcclient.InferResult(result.get_response())
        seq_len = infer_result.as_numpy("sequence_length")[0, 0]
        # token = np.array([infer_result.as_numpy("output_ids")[0, 0, seq_len-1]])
        # print("infer_result shape:{}".format(infer_result.as_numpy("output_ids").shape))
        
        output_ids = infer_result.as_numpy("output_ids")[0][0]
        # print("output_ids:{}".format(output_ids))
        
        # token = np.array(output_ids)
        token_text = tokenizer.decode(output_ids)
        # print("result:{}".format(result))
        print("seq_len:{} token_text:{}".format(seq_len, token_text), end="\n")
        
        print("end innner:", time.time())
        
    sys.stdout.flush()


def inference(FLAGS, lines):
    # invoking-loop
    index = 0
    batch_size = 1
    torch.set_printoptions(precision=6)
    tokenizer = LlamaTokenizer.from_pretrained(FLAGS.hf_model_location, cache_dir="models", padding_side='right')
    request_parallelism = 1
    # debug for triton
    verbose = 0
    test = False
    
    result_queue = mp.Queue()
    consumer = mp.Process(target=stream_consumer, args=(result_queue, False))
    consumer.start()
    
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=request_parallelism,
                                        verbose=verbose) as client:
        while index < len(lines):
            # build request
            input_text = lines[index]
            print("======> input_text:{}".format(input_text))
            
            inputs = _token2inputs(FLAGS, tokenizer(input_text, return_tensors='pt', padding=True))
            print(f"\033[0;30;41mPrompt\033[0;31m {input_text} ")

            # rpc-invoke
            # Create gRPC stub for communicating with the server
            print(f"\033[0;30;42mRobot\033[0;32m", end="")  # without "\n"

            # async stream infer(callback will print it)
            begin = time.time()
            print("begin:", begin)
            
            client.start_stream(callback=partial(stream_callback, result_queue, tokenizer))
            result_queue.put(time.perf_counter())
            client.async_stream_infer(FLAGS.model_name, inputs)
            result_queue.put(None)
            
            consumer.join()
            
            end = time.time()
            
            print("time cost:{}".format(end - begin))

            print(f"\033[0m")

            # update index
            index += batch_size
            
            # debug
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='127.0.0.1:8001',
                        help='Inference server Appkey. Default is .')
    parser.add_argument('-pro', '--protocol', type=str, required=False, default='grpc',
                        help='Inference server Appkey. Default is .')
    parser.add_argument('--hf_model_location', type=str,
                        default="/mnt/data/atomgpt/model_sft_atomgpt_28000_0613/checkpoint-12000_merge",
                        help="tokenizer model path")
    parser.add_argument('-max_output_len', '--maximum_output_length', type=int, default=128, metavar='NUMBER',
                        help='maximum output length (default: 128)')
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='Beam width for beam search. If setting 1, then using sampling.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=50, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.95, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('--model_name', type=str, default="fastertransformer",
                        help='model_name')
    FLAGS = parser.parse_args()

    def generate_prompt(input_text):
        prompt = ''
        prompt +=  "<s>Human: "+input_text.strip()+"\n</s><s>Assistant: "
        return prompt
    
    # Infer
    head_lines = [generate_prompt("北京有什么？")]
    for row in reader:
        head_lines.append(row[0])
    inference(FLAGS, head_lines)

