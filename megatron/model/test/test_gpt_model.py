import unittest
from unittest.mock import patch

from megatron.model import GPTModel

def default_args():
    """return a dictionary with key as argument name and value as additional arguments"""
    VOCAB_FILE=""
    MERGE_FILE=""

    CHECKPOINT_PATH=""
    DATA_PATH=""

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
        "--lr-decay-style": "cosine",
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
        "--prefix-lm": "",
        "--reset-attention-mask": "",

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

class MyTestCase(unittest.TestCase):
    def test_gpt_causal(self):
        """Test causal invariance, ie past token don't depend on future tokens."""
        with patch('sys.argv'. ['stuff']):

        model = GPTModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=True,
            post_process=True
        )
        self.assertEqual(True, False)  # add assertion here

    def test_gpt_prefix(self):
        """Test prefix invariance, ie past tokens in the target don't depend on future tokens."""


if __name__ == '__main__':
    unittest.main()
