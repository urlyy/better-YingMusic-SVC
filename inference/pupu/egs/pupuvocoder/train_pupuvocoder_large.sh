######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $exp_dir))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

export TORCH_DISABLE_ADDR2LINE=1

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,name:,checkpoint:,resume_type:,main_process_port: -- "$@")
eval set -- "$options"

while true; do
  case $1 in
    # Experimental Name
    -n | --name) shift; exp_name=$1 ; shift ;;
    # Visible GPU machines. The default value is "0".
    --gpu) shift; gpu=$1 ; shift ;;

    # [Only for Training] The specific checkpoint path that you want to resume from.
    --checkpoint) shift; checkpoint=$1 ; shift ;;
    # [Only for Training] `resume` for loading all the things (including model weights, optimizer, scheduler, and random states). `finetune` for loading only the model weights.
    --resume_type) shift; resume_type=$1 ; shift ;;
    # [Only for Traiing] `main_process_port` for multi gpu training
    --main_process_port) shift; main_process_port=$1 ; shift ;;

    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

if [ -z "$gpu" ]; then
    gpu="0"
fi

if [ -z "$main_process_port" ]; then
    main_process_port=29500
fi
echo "Main Process Port: $main_process_port"

######## Training ###########
echo "Exprimental Name: $exp_name"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --main_process_port "$main_process_port" \
    --mixed_precision="no" \
    "${work_dir}"/bins/train.py \
    --config "${work_dir}"/egs/pupuvocoder/exp_config_pupuvocoder_large.json \
    --exp_name "$exp_name" \
    --log_level info \
    --checkpoint "$checkpoint" \
    --resume_type "$resume_type"