# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining. It's supposed to be a faster version compared to vanilla preprocess.py"""

import argparse
import collections
import itertools
import json
import multiprocessing
import os
import sys
import threading
from multiprocessing.connection import Connection

from megatron.data.indexed_dataset import index_file_path, data_file_path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.json_keys = args.json_keys
        self.append_eod = args.append_eod
        self.file = open(args.input, 'r')
        # Use Encoder class as a container for global data
        self.tokenizer = build_tokenizer(args)
        if args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                self.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                self.splitter = splitter

        else:
            self.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in self.splitter.tokenize(text):
                sentence_ids = self.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.append_eod:
                doc_ids[-1].append(self.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)

    def get_json_lines(self, chunk_segment):
        # We know chunk_segment represents a few lines
        start, end = chunk_segment
        self.file.seek(start)
        return self.file.readlines(end-start)


def process_samples(simple_queue, process_id, args, level, writer: Connection):
    encoder = Encoder(args)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_filename = get_output_filename(args.output_prefix, key, level, process_id)
        output_bin_files[key] = data_file_path(output_filename)
        output_idx_files[key] = index_file_path(output_filename)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                     impl=args.dataset_impl,
                                                     vocab_size=encoder.tokenizer.vocab_size)

    chunk_segment = simple_queue.get()
    while chunk_segment is not None:
        process_chunk_segment(chunk_segment, encoder, builders, writer)

        chunk_segment = simple_queue.get()

    # In case finished, we still need to add None to signal to everyone else
    simple_queue.put(None)
    # Send None as end of sequence signal
    writer.send((None, process_id))
    writer.close()

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

    print(f"Worker {process_id} finished", flush=True)


def process_chunk_segment(chunk_segment, encoder, builders, writer):
    total_bytes_processed = 0
    json_lines = encoder.get_json_lines(chunk_segment)
    for json_line in json_lines:
        if json_line.strip() == "":
            continue

        doc, bytes_processed = encoder.encode(json_line)

        total_bytes_processed += bytes_processed

        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()

    writer.send((len(json_lines), total_bytes_processed))


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None, 
                       help="Name or path of the huggingface tokenizer.")

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def fill_simple_queue(filename, simple_queue, chunk_size:int):
    # TODO: Assess if instead we could feed pointers which process can then load.
    with open(filename, "r") as f:
        print("Start filling queue", flush=True)
        start = f.tell()
        while True:
            empty_chunk = True
            for _ in range(chunk_size):
                line = f.readline()
                if line == "":
                    break
                empty_chunk = False
            if empty_chunk:
                simple_queue.put(None)
                print(f"Finished reading input file", flush=True)
                return
            end = f.tell()
            simple_queue.put((start, end))
            start = end

def log(readers, log_interval):
    print("Start Logging", flush=True)
    proc_start = time.time()
    total_bytes_processed = 0
    doc_processed = 0
    logged_docs = 0

    # we want to compute a rolling average of bytes processed over last 10k documents (more or less)
    bytes_queue_max_length = 10_000 // log_interval + 1
    bytes_queue = collections.deque(maxlen= bytes_queue_max_length)
    # we fill the queue with (start_time, 0)
    bytes_queue.extend([(proc_start, total_bytes_processed)]*bytes_queue_max_length)

    while len(readers) != 0:
        for r in multiprocessing.connection.wait(readers):
            # Can be:
            #  - tuple (bytes: int, nb_of_docs): When process notify the writer that
            #  - tuple (None, process_index): When process finish their processing of data.
            data = r.recv()
            if data[0] is None:
                process_index = data[1]
                # This means that a worker has finished.
                r.close()
                readers.remove(r)
                print(f"Process {process_index} finished working. Remaining workers: {len(readers)}", flush=True)
                continue

            nb_of_docs, bytes_processed = data
            total_bytes_processed += bytes_processed
            doc_processed += nb_of_docs

            if (doc_processed - logged_docs) >= log_interval:
                logged_docs = doc_processed
                current = time.time()
                elapsed = current - proc_start

                (old_start_time, old_bytes) = bytes_queue.popleft()
                bytes_queue.append((current, total_bytes_processed))
                mbs = (total_bytes_processed - old_bytes) / (current - old_start_time) / 1024 / 1024
                print(f"Processed {doc_processed} documents",
                      f"({doc_processed / elapsed} docs/s, {mbs} MB/s).", flush=True)


def get_output_filename(prefix, key, level, process_index = None):
    if process_index is None:
        return f"{prefix}_{key}_{level}"
    else:
        return f"{prefix}_{key}_{level}_{process_index}"

def main():
    args = get_args()

    print("Opening", args.input)
    simple_queue = multiprocessing.Queue(1_000) # we can also limit the number of elements to reduce the memory footprint.
    chunk_size = 25

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    assert args.workers > 1, "One for filling the queue"
    readers, writers = list(zip(*[multiprocessing.Pipe(duplex=False) for _ in range(args.workers - 1)]))
    process_ids = list(range(len(writers)))
    processes = [multiprocessing.Process(target=process_samples, args=(simple_queue, process_id, args, level, writer)) for process_id, writer in zip(process_ids, writers)]
    log_thread = threading.Thread(target=log, args=(list(readers), args.log_interval))
    fill_thread = multiprocessing.Process(target=fill_simple_queue, args=(args.input, simple_queue, chunk_size))

    fill_thread.start()
    log_thread.start()
    for i, process in enumerate(processes):
        process.start()

    # We close the writable end of the pipe now to be sure that
    # p is the only process which owns a handle for it.  This
    # ensures that when p closes its handle for the writable end,
    # wait() will promptly report the readable end as being ready.
    # https://docs.python.org/fr/3/library/multiprocessing.html#multiprocessing.connection.Connection
    for writer in writers:
        writer.close()

    fill_thread.join()
    fill_thread.close()
    for process in processes:
        process.join()
        process.close()
    log_thread.join() #TODO: figure out why there seems to be a possible dead lock situation.

    # TODO: this may be done after.
    print("Merging files together", flush=True)

    tokenizer = build_tokenizer(args)

    print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
    print(f"Output prefix: {args.output_prefix}", flush=True)
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_filename = f"{args.output_prefix}_{key}_{level}"
        output_bin_files[key] = data_file_path(output_filename)
        output_idx_files[key] = index_file_path(output_filename)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                     impl=args.dataset_impl,
                                                     vocab_size=tokenizer.vocab_size)

    for key in args.json_keys:
        for process_id in process_ids:
            output_filename = get_output_filename(args.output_prefix, key, level, process_id)
            builders[key].merge_file_(output_filename)
        builders[key].finalize(output_idx_files[key])

    # Remove temporary files
    print("Removing shard files")
    for key in args.json_keys:
        for process_id in range(len(processes)):
            output_filename = get_output_filename(args.output_prefix, key, level, process_id)
            os.remove(data_file_path(output_filename))
            os.remove(index_file_path(output_filename))

if __name__ == '__main__':
    main()
