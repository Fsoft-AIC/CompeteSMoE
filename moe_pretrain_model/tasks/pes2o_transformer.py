from .lm_base import LMBase
from framework.task import task, args
from framework import dataset
import framework
from .lm_eval_mixin import LMEvalMixin


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-sentencepiece.n_pieces", default=8000)
    parser.add_argument("-lmds.valid_ratio", default=1.0)


@task()
class Pes2oTransformer(LMEvalMixin, LMBase):
    helper: framework.helpers.TrainingHelper

    def create_datasets(self):
        self.batch_dim = 1
        # print(self.helper.args.stop_after)
        # breakpoint()
        if self.helper.args.stop_after is not None:
            train_token_limit = self.helper.args.lm.unroll * self.helper.args.batch_size * (self.helper.args.stop_after + 100)
        else:
            train_token_limit = None

        self.train_set = dataset.PES2O(
            self.helper.args.lm.unroll, split="train", n_tokens=self.helper.args.sentencepiece.n_pieces,
            token_limit=train_token_limit)

        # Magic number for backward compatibility
        test_token_limit = int((842021184 // self.helper.args.lm.unroll) * self.helper.args.lmds.valid_ratio) * self.helper.args.lm.unroll

        self.valid_sets.val = dataset.PES2O(
            self.helper.args.lm.unroll, split="validation", n_tokens=self.helper.args.sentencepiece.n_pieces,
            token_limit=test_token_limit)

        super().create_datasets()
