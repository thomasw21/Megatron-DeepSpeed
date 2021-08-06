import unittest
from random import randint
from unittest.mock import patch

import torch

from megatron import initialize_megatron, get_args, get_tokenizer
from megatron.model import GPTModelPipe
from pretrain_gpt import get_batch_pipe


def get_default_args():
    """return a dictionary with key as argument name and value as additional arguments"""
    VOCAB_FILE=""
    MERGE_FILE=""

    CHECKPOINT_PATH=""
    DATA_PATH=""

    return {
        # Deepspeed
        "--deepspeed": "",

        # GPT_ARGS
        "--num-layers": "2",
        "--hidden-size": "128",
        "--num-attention-heads": "4",
        "--seq-length": "256",
        "--max-position-embeddings": "256",
        "--micro-batch-size": "4",
        "--global-batch-size": "8",
        "--lr-decay-iters": "320000",
        "--lr-decay-style": "cosine",
        "--lr": "0.00015",
        "--min-lr": "1.0e-5",
        "--train-iters": "5000",
        "--vocab-file": VOCAB_FILE,
        "--merge-file": MERGE_FILE,
        "--data-impl": "mmap",
        "--split": "949,50,1",
        "--distributed-backend": "nccl",
        "--weight-decay": "1e-2",
        "--clip-grad": "1.0",
        "--lr-warmup-fraction": ".01",
        "--fp16": "",

        # OUTPUT_ARGS
        "--log-interval": "10",
        "--save-interval": "500",
        "--eval-interval": "100",
        "--eval-iters": "10",
        "--checkpoint-activations": "",

        # DATA_ARGS
        "--save": CHECKPOINT_PATH,
        "--load": CHECKPOINT_PATH,
        "--data-path": DATA_PATH,
    }

def flatten_arguments(args):
    """
    Converts dictionary argument to a list

    Example: {"arg1": "value1", "arg2": "value2"} -> ["arg1", "value1", "arg2", "value2"]
    """
    return [item for key_value in args.items() for item in key_value]

class MyTestCase(unittest.TestCase):
    def test_gpt_causal(self):
        """Test causal invariance, ie past token don't depend on future tokens."""
        command_args = get_default_args()

        with patch('sys.argv', flatten_arguments(command_args)):
            initialize_megatron()
            args = get_args()
            tokenizer = get_tokenizer()

            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
            )

            token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

            # eod is a special token
            token_ids[token_ids == tokenizer.eod] += 1
            token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

            # we set a variation on the inputs
            changed_index = randint(0, args.seq_length - 1)
            token_ids_changed = token_ids.clone()
            token_ids_changed[changed_index] = (token_ids_changed[changed_index] + 1) % args.padded_vocab_size

            model.forward()[get_batch_pipe(token_ids)]

            position_ids = torch.arange(args.seq_length).unsqueeze(0)
            attention_mask = torch.ones((args.micro_batch_size, 1, args.seq_length, args.seq_length))

            output = model(input_ids, position_ids, attention_mask)[0]



            output_changed = model(input_ids_changed, position_ids, attention_mask)[0]

            # All token in past should be unchanged
            self.assertEqual(output[:, :changed_index], output_changed[:, :changed_index])


    def test_gpt_prefix(self):
        """Test prefix invariance, ie past tokens in the target don't depend on future tokens."""
        command_args = get_default_args()

        command_args["--prefix-lm"] = "",
        command_args["--reset-attention-mask"] = "",

        with patch('sys.argv', flatten_arguments(command_args)):
            initialize_megatron()
            args = get_args()

            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=True,
                post_process=True
            )

            input_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))
            position_ids = torch.arange(args.seq_length).unsqueeze(0)
            attention_mask = torch.ones((args.micro_batch_size, 1, args.seq_length, args.seq_length))

            output = model(input_ids, position_ids, attention_mask)[0]

            changed_index = randint(0, args.seq_length - 1)
            input_ids_changed = input_ids.clone()
            input_ids_changed[changed_index] = (input_ids_changed[changed_index] + 1) % args.padded_vocab_size

            output_changed = model(input_ids_changed, position_ids, attention_mask)[0]

            # All token in past should be unchanged
            self.assertEqual(output[:, :changed_index], output_changed[:, :changed_index])

    def test_gpt_rotary_embeddings(self):
        """Test rotary embeddings"""
        command_args = get_default_args()

        del command_args["--max-position-embeddings"]
        command_args["--position-embedding-type"] = "rotary"

        with patch('sys.argv', flatten_arguments(command_args)):
            initialize_megatron()
            args = get_args()

            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=True,
                post_process=True
            )

            input_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))
            position_ids = torch.arange(args.seq_length).unsqueeze(0)
            attention_mask = torch.ones((args.micro_batch_size, 1, args.seq_length, args.seq_length))

            output = model(input_ids, position_ids, attention_mask)[0]

            changed_index = randint(0, args.seq_length - 1)
            input_ids_changed = input_ids.clone()
            input_ids_changed[changed_index] = (input_ids_changed[changed_index] + 1) % args.padded_vocab_size

            output_changed = model(input_ids_changed, position_ids, attention_mask)[0]

            # All token in past should be unchanged
            self.assertEqual(output[:, :changed_index], output_changed[:, :changed_index])


if __name__ == '__main__':
    unittest.main()
