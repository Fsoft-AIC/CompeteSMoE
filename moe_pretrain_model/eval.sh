export CUDA_VISIBLE_DEVICES="0"
#!/bin/bash
export API_TYPE='xxxx'
export DEPLOYMENT='xxx'
export ENDPOINT='xxxxx'
export VERSION='xxxx'
export API_KEY='xxxx'
export KEY_HF="xxxxx"
export CUDA_LAUNCH_BLOCKING=0
export HF_HOME="/cm/shared/anonymous_H102/toolkitmoe/evaluate"
export TMPDIR="/cm/shared/anonymous_H102/tmp"
export TOOLKIT_DIR="/cm/shared/anonymous_H102"  # Path to the toolkitmoe directory
export PYTHONPATH="/cm/shared/anonymous_H102/toolkitmoe/moe_pretrain_model":$PYTHONPATH
export tensorboard="/home/anonymous/miniconda3/envs/moe/lib/python3.9/site-packages/tensorboard"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Set the master address (địa chỉ của máy chủ master, sử dụng IP của máy chủ master hoặc localhost)
MASTER_ADDR="127.0.0.3"  # Ví dụ với localhost, thay đổi theo yêu cầu

# Set the master port (cổng mặc định là 12345, có thể thay đổi nếu cần)
MASTER_PORT=12313

# Set the environment variable for PORT if needed (nếu không sử dụng giá trị mặc định trong code)
export MASTER_PORT=$MASTER_PORT
cd /cm/shared/anonymous_H102/toolkitmoe/moe_pretrain_model
# Run the distributed training using torch.distributed.run (thay vì torch.distributed.launch)
python /cm/shared/anonymous_H102/toolkitmoe/moe_pretrain_model/paper/moe_universal/run_tests.py
# python3 main.py \
#     --name post_validate \
#     --restore /cm/archive/anonymous/checkpoints/safe_pretrain_final/not_ut_smoe/slimpajama_moe_no_attmoe_154M_2/checkpoint/model-100000.pth \
#     --test_only 1 \
#     -reset 1 \
#     -lm.eval.enabled 1 \
#     --lm.eval.ai2arc.enabled 1\
#     --keep_alive 0 \
#     --batch_size 2 \
#     --save_name_logs post_train
# bash /cm/shared/anonymous_H102/toolkitmoe/imagenet64/train2.sh
# python /cm/shared/anonymous_H102/toolkitmoe/moe_pretrain_model/analyst/process_data.py