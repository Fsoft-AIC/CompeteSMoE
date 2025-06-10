import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ['TORCH_LOGS'] = "+dynamo"
# os.environ['TORCHDYNAMO_VERBOSE'] = "1"

from typing import Optional
import framework
from framework.task import task_db
import torch
import json
from framework import dataset
import tasks
import random
import numpy as np
from layers.moe import MoE
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import json
import os

def save_json(data, file_path):
    """
    Save a dictionary as a JSON file.

    :param data: Dictionary to save
    :param file_path: Path to save the JSON file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write JSON file
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)  # Support UTF-8 characters

        print(f"JSON file saved at: {file_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
def register_args(parser: framework.helpers.ArgumentParser):
    task_db.register_args(parser)
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-embedding_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.n_heads", default=4)
    parser.add_argument("-transformer.variant", default="standard")
    parser.add_argument("-transformer.ff_multiplier", default=2.0)
    parser.add_argument("-transformer.encoder_n_layers", default=3)
    parser.add_argument("-transformer.attention_dropout", default=0.0)
    parser.add_argument("-load_pretrained_model", type=str)
    parser.add_argument("-test_pretrained", default=1)
    parser.add_argument("-train_baseline", default=False, help="Train the model on easy task and test on hard,"
                                                               "no masking")
    parser.add_argument("-test_only", default=False)
    parser.add_argument("-save_name_logs", default="results")
    parser.add_argument("-nan_detect", default=False)
    parser.add_argument("-fs_cache_pattern", default="*", parser=parser.str_or_none_parser)


def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(wandb_project_name="lm",
                                              register_args=register_args, extra_dirs=["export", "model_weights", "tmp"],
                                              log_async=True, restore=restore)

    dataset.init_fs_cache(helper.args.fs_cache_pattern)
    # breakpoint()
    task = task_db.get_task(helper.args.task)

    # breakpoint()
    task = task(helper)
    # print("++++++++++++++++++")
    # print(task)
    # breakpoint()
    return helper, task

def main():
    helper, task = initialize()
    set_seed(42)
    if helper.args.nan_detect:
        torch.autograd.set_detect_anomaly(True)

    if helper.args.load_pretrained_model:
        assert not helper.args.train_baseline

        print("Loading pretrained model...")

        pretrained = os.path.expanduser(helper.args.load_pretrained_model)
        if not helper.args.load_pretrained_model.endswith(".pth"):
            pretrained = os.path.join(pretrained, str(helper.args.sweep_id_for_grid_search), "model.pth")

        assert os.path.isfile(pretrained), f"Failed to load pretrained weights. File {pretrained} not found."

        if helper.dist_env.is_master():
            task.load_weights(pretrained)

        helper.distibute_model_weights()
        print("Done.")
    print(helper.args.task)
    if helper.args.test_only:
        res = task.validate()
        path_save = os.path.dirname(helper.args.restore)
        assert helper.args.restore is not None,  "Warning: Restore must to not None"
        # if helper.args.restore:
        name_checkpoint = os.path.basename(helper.args.restore).split('.')[0]
        # result_experts = {}
        # for id, layer in enumerate(task.model_interface.model):
        #     for module in layer.modules():
        #         if isinstance(module, MoE): 
        #             if hasattr(module, "get_dist_experts"):
        #                 result_experts[module.name_moe] = module.dist_experts.tolist()
                      
        
        helper.log(res)
      
        save_json(data=res, file_path=os.path.join(path_save, f"{helper.args.save_name_logs}/{helper.args.task}_{name_checkpoint}.json"))
        print("Validate returned:")
        print(json.dumps(res))
        print("-------------------")
    else:
        if helper.args.test_pretrained and helper.args.load_pretrained_model:
            helper.log({f"load_validation/{k}": v for k, v in task.validate().items()})

        if helper.args.train_baseline:
            task.set_baseline_mode()

        task.train()

        print("Training finished. Saving model...")
        task.save_weights()

    task.finish()
    helper.finish()


if __name__ == "__main__":
    main()
