import os
import lib
import json
import copy
from typing import List, Optional
from tqdm import tqdm


my_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(my_dir+"/../..")
my_rel_dir = os.path.relpath(my_dir, main_dir)
curr_dir = os.getcwd()

result_name ="result_competition"
TESTS = "-lm.eval.lambada.enabled 1 -lm.eval.cbt.enabled 1 -lm.eval.hellaswag.enabled 1 -lm.eval.piqa.enabled 1 -lm.eval.blimp.enabled 1 -lm.eval.ai2arc.enabled 1"


def get_info(id, patch_ckpt=None, bs: Optional[int] = None):
    dest_dir = f"{id}/"
    path_checkpoint = os.path.basename(patch_ckpt).split('.')[0]
    res_path = f"{dest_dir}{result_name}_{path_checkpoint}.json"

    if not os.path.isfile(res_path) or True:
        # ckpt_path = f"{dest_dir}/model-10000.ckpt"
        # os.chdir(main_dir)

        # ckpt_path = f"{my_rel_dir}/{ckpt_path}"
        if patch_ckpt is not None:
            ckpt_path = patch_ckpt

        if bs is None:
            bs = ""
        else:
            bs = f"--batch_size {bs}"

        cmd = f"python3 main.py --name post_validate --restore {ckpt_path} --test_only 1 -reset 1 -lm.eval.enabled 1 {TESTS} --keep_alive 0 {bs}"
        print("Validate command: ", cmd)
        out = lib.run_command(cmd)
        lines = out.splitlines()
        start_line = lines.index('Validate returned:')
        end_line = None
        for i in range(start_line, len(lines)):
            if lines[i].startswith("-------"):
                end_line = i
                break

        assert end_line is not None

        res = "\n".join(lines[start_line+1:end_line])
        os.chdir(curr_dir)

        with open(res_path, "w") as f:
            f.write(res)

    with open(res_path, "r") as f:
        res = json.load(f)

    return res


if __name__ == "__main__":
    
    # runs = lib.get_runs([
    #     "c4_baseline_big_rope_nodrop", "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    #     "c4_baseline_bigger_rope_nodrop", "c4_moeut_bigger_matched_rope_nonorm_g3",
    #     "c4_baseline_small_rope_long_nodrop", "c4_moeut_small_matched_rope_noln_long",
    #     "c4_moeut_gigantic_switchead", "c4_baseline_gigantic_rope_nodrop",
    #     "c4_moeut_mid_matched_rope_noln", "c4_baseline_mid_rope_nodrop",
    #     "c4_moeut_1b_switchead", "c4_baseline_1b_rope_nodrop",

    #     "pes2o_moeut_small_matched_rope_noln_long",
    #     "pes2o_baseline_small_rope_long_nodrop",
    #     "pes2o_baseline_big_rope_nodrop", "pes2o_moeut_big_matched_rope_noln",

    #     "slimpajama_baseline_small_rope_long_nodrop", "slimpajama_baseline_big_rope_nodrop",
    #     "slimpajama_moeut_small_matched_rope_noln_long", "slimpajama_moeut_big_matched_rope_noln",
    #     "slimpajama_moeut_1b_switchead", "slimpajama_baseline_1b_rope_nodrop",
    # ])
    # runs = [r for r in runs if not os.path.isfile(f"checkpoints/{r.id}/{result_name}.json")]
    # for r in tqdm(runs):
    #     get_info(r.id, bs=16)
    list_eval = [
        # (
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_smoeut_norm/slimpajama_moe_no_attmoe_154M/checkpoint", 
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_smoeut_norm/slimpajama_moe_no_attmoe_154M/checkpoint/model-100000.pth", 
        #     8
        # ),
        # (
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_xmoe_origin/slimpajama_moe_no_attmoe_154M/checkpoint", 
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_xmoe_origin/slimpajama_moe_no_attmoe_154M/checkpoint/model-100000.pth", 
        #     8
        # ),
        # (
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_competesmoe10/slimpajama_moe_no_attmoe_154M/checkpoint", 
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_competesmoe10/slimpajama_moe_no_attmoe_154M/checkpoint/model-20000.pth", 
        #     25
        # ),
        # (
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_competesmoe10_CS_BS_theta0.2_RL0.001/slimpajama_moe_no_attmoe_154M/checkpoint", 
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_competesmoe10_CS_BS_theta0.2_RL0.001/slimpajama_moe_no_attmoe_154M/checkpoint/model-10000.pth", 
        #     20
        # ),
        # (
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_competesmoe13_SIG/slimpajama_moe_no_attmoe_154M_competesmoe/checkpoint", 
        #     "/cm/archive/anonymous/checkpoints/pretrain/not_ut_competesmoe13_SIG/slimpajama_moe_no_attmoe_154M_competesmoe/checkpoint/model-10000.pth", 
        #     8
        # ),
        # (
        #     "/cm/archive/anonymous/checkpoints/pretrain/Xnot_ut_competesmoe13_SIG_not_SIG_in_COMP/slimpajama_moe_no_attmoe_154M_competesmoe/checkpoint", 
        #     "/cm/archive/anonymous/checkpoints/pretrain/Xnot_ut_competesmoe13_SIG_not_SIG_in_COMP/slimpajama_moe_no_attmoe_154M_competesmoe/checkpoint/model-10000.pth", 
        #     8
        # ),
        
        # (
        #     "/cm/archive/anonymous/checkpoints/pretrain_final/not_ut_competesmoe_final_theta0.4/slimpajama_moe_no_attmoe_154M_competesmoe/checkpoint/", 
        #     "/cm/archive/anonymous/checkpoints/pretrain_final/not_ut_competesmoe_final_theta0.4/slimpajama_moe_no_attmoe_154M_competesmoe/checkpoint/model-100000.pth", 
        #     30
        # ),
        (
            "/cm/archive/anonymous/checkpoints/safe_pretrain_final/not_ut_smoe/slimpajama_moe_no_attmoe_154M_2/checkpoint/", 
            "/cm/archive/anonymous/checkpoints/safe_pretrain_final/not_ut_smoe/slimpajama_moe_no_attmoe_154M_2/checkpoint/model-100000.pth", 
            20
        ),
        
        # (
        #     "/cm/archive/anonymous/checkpoints/safe_pretrain_final/not_ut_deepseekv2_1share/slimpajama_moe_no_attmoe_154M_deepseekv2/checkpoint/", 
        #     "/cm/archive/anonymous/checkpoints/safe_pretrain_final/not_ut_deepseekv2_1share/slimpajama_moe_no_attmoe_154M_deepseekv2/checkpoint/model-100000.pth", 
        #     40
        # ),
        
       


    ]
    for id, path_weight, bs  in list_eval:
        get_info(id, path_weight, bs)
    # breakpoint()