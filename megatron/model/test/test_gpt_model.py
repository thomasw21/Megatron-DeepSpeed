import unittest
from random import randint
from unittest.mock import patch

import torch
from deepspeed import deepspeed

from megatron import initialize_megatron, get_args, get_tokenizer
from pretrain_gpt import model_provider as gpt_model_provider, get_batch_pipe as get_gpt_batch_pipe
from pretrain_prefix_lm import model_provider as prefix_lm_model_provider, get_batch_pipe as get_prefix_lm_batch_pipe

def get_default_args():
    """return a dictionary with key as argument name and value as additional arguments"""
    return {
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
        "--tokenizer-type": "PretrainedFromHF",
        "--tokenizer-name-or-path": "gpt2",
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
    }

def flatten_arguments(args):
    """
    Converts dictionary argument to a list.

    Note: we add "IGNORED" at the beginning as this value is ignored by the argparser

    Example: {"arg1": "value1", "arg2": "value2"} -> ["IGNORED", "arg1", "value1", "arg2", "value2"]
    """
    return ["IGNORED"] + [item for key_value in args.items() for item in key_value if item != ""]

class MyTestCase(unittest.TestCase):
    def setUpClass(cls) -> None:
        deepspeed.init_distributed()

    def tearDown(self) -> None:
        # We reset all global variables
        global _GLOBAL_ARGS
        global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
        global _GLOBAL_TOKENIZER
        global _GLOBAL_TENSORBOARD_WRITER
        global _GLOBAL_ADLR_AUTORESUME
        global _GLOBAL_TIMERS

        _GLOBAL_ARGS = None
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
        _GLOBAL_TOKENIZER = None
        _GLOBAL_TENSORBOARD_WRITER = None
        _GLOBAL_ADLR_AUTORESUME = None
        _GLOBAL_TIMERS = None

    def test_gpt_causal(self):
        """Test causal invariance, ie past token don't depend on future tokens."""
        command_args = get_default_args()

        with patch('sys.argv', flatten_arguments(command_args)):
            initialize_megatron()
            args = get_args()
            tokenizer = get_tokenizer()

            model_engine = deepspeed.init_inference(gpt_model_provider())

            token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

            # eod is a special token
            token_ids[token_ids == tokenizer.eod] += 1
            token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

            # process batch
            input_batch = get_gpt_batch_pipe(token_ids)[0]

            # get a modified version of the first batch, we change a specific index
            changed_index = randint(0, args.seq_length - 2)
            input_token_ids_changed = input_batch[0].clone()
            # We increment the token_id by one for that index in order to artificially change the sequence.
            input_token_ids_changed[changed_index] = (input_token_ids_changed[:, changed_index] + 1) % args.padded_vocab_size

            output = model_engine(input_batch)
            output_changed = model_engine((input_token_ids_changed, *input_batch[1:]))

            # All token in past should be unchanged
            self.assertTrue(
                torch.all(
                    output[:, :changed_index].eq(output_changed[:, :changed_index])
                )
            )
            # All tokens in the future should have changed
            self.assertFalse(
                torch.any(
                    output[:, changed_index:].eq(output_changed[:, changed_index:])
                )
            )


    def test_gpt_prefix(self):
        """
        Test prefix invariances:
            - Past target tokens don't depend on future target tokens.
            - Target tokens depend on input tokens.
            - Input tokens depend on all other input tokens, but never target tokens.
        """
        command_args = get_default_args()

        command_args["--prefix-lm"] = "",
        command_args["--reset-attention-mask"] = "",

        with patch('sys.argv', flatten_arguments(command_args)):
            initialize_megatron()
            args = get_args()
            tokenizer = get_tokenizer()

            model_engine = deepspeed.init_inference(prefix_lm_model_provider())

            token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

            # eod is a special token, this also guarantees that the whole row is considered as a document.
            token_ids[token_ids == tokenizer.eod] += 1
            token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

            # process batch to have non empty prefix
            for i in range(9, -1, -1):
                input_batch, _, prefix_indices = get_prefix_lm_batch_pipe(token_ids)
                if (prefix_indices[0][0] != 0):
                    break
                if i == 0:
                    # FIXME: find a better way to not obtain empty prefix
                    raise ValueError("Could not obtain non pathological case where prefix is not empty")

            output = model_engine(input_batch)

            ## --------------- CHANGE A TARGET TOKEN ---------------------------
            # get a modified version of the first batch
            changed_target_index = prefix_indices[0][0] # guaranteed to exist as each row has at least one partial document
            token_ids_changed_target = input_batch[0].clone()
            # We increment the token id on the changed index.
            token_ids_changed_target[changed_target_index] = (token_ids_changed_target[0, changed_target_index] + 1) % args.padded_vocab_size
            # make sure we're not changing a token to eod as it's a special token
            token_ids_changed_target[token_ids_changed_target == tokenizer.eod] += 1
            token_ids_changed_target[token_ids_changed_target == tokenizer.eod] %= args.padded_vocab_size

            # Test change
            output_changed_target = model_engine((token_ids_changed_target, *input_batch[1:]))

            # All token in past should be unchanged
            self.assertTrue(
                torch.all(
                    output[0, :changed_target_index].eq(output_changed_target[0, :changed_target_index])
                )
            )
            # All tokens in the future should have changed
            self.assertFalse(
                torch.any(
                    output[0, changed_target_index:].eq(output_changed_target[0, changed_target_index:])
                )
            )
            # Unchanged changed rows should not change either
            self.assertTrue(
                torch.all(
                    output[1, :].eq(output_changed_target[1, :])
                )
            )

            ## --------------- CHANGE AN INPUT TOKEN ---------------------------
            # Let's change the the last prefix token and make sure that the first token changed
            last_prefix_index = prefix_indices[0][0] - 1  # guaranteed to be positive as we avoid pathological case previously
            token_ids_changed_input = input_batch[0].clone()
            #  We increment the token id on the changed index.
            token_ids_changed_input[changed_target_index] = (token_ids_changed_input[
                                                                 0, last_prefix_index] + 1) % args.padded_vocab_size
            # make sure we're not changing a token to eod as it's a special token
            token_ids_changed_input[token_ids_changed_input == tokenizer.eod] += 1
            token_ids_changed_input[token_ids_changed_input == tokenizer.eod] %= args.padded_vocab_size

            output_changed_input = model_engine((token_ids_changed_input, *input_batch[1:]))

            # All tokens should be changed
            self.assertFalse(
                torch.any(
                    output[0, :].eq(output_changed_input[0, :])
                )
            )
            # Unchanged changed rows should not change either
            self.assertTrue(
                torch.all(
                    output[1, :].eq(output_changed_input[1, :])
                )
            )


    def test_gpt_rotary_embeddings(self):
        """Test rotary embeddings"""
        command_args = get_default_args()

        del command_args["--max-position-embeddings"]
        command_args["--position-embedding-type"] = "rotary"

        with patch('sys.argv', flatten_arguments(command_args)):
            initialize_megatron()
            args = get_args()
            tokenizer = get_tokenizer()

            model_engine = deepspeed.init_inference(gpt_model_provider())

            token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

            # eod is a special token
            token_ids[token_ids == tokenizer.eod] += 1
            token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

            # process batch
            input_batch = get_gpt_batch_pipe(token_ids)[0]

            model_engine(input_batch)

if __name__ == '__main__':
    unittest.main()
