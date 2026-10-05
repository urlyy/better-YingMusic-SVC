######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $exp_dir))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

export TORCH_DISABLE_ADDR2LINE=1

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,infer_datasets:,infer_expt_dir:,infer_output_dir: -- "$@")
eval set -- "$options"

while true; do
  case $1 in
    # Visible GPU machines. The default value is "0".
    --gpu) shift; gpu=$1 ; shift ;;

    # [Only for Inference] The inferenced datasets
    --infer_datasets) shift; infer_datasets=$1 ; shift ;;
    # [Only for Inference] The experiment dir. The value is like "[Your path to save logs and checkpoints]/[YourExptName]"
    --infer_expt_dir) shift; infer_expt_dir=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios. Its default value is "$expt_dir/result"
    --infer_output_dir) shift; infer_output_dir=$1 ; shift ;;

    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

if [ -z "$gpu" ]; then
    gpu="0"
fi

if [ -z "$infer_output_dir" ]; then
    infer_output_dir="$infer_expt_dir/result"
fi

######## Inference ###########
CUDA_VISIBLE_DEVICES=$gpu accelerate launch "${work_dir}"/bins/inference.py \
    --config "${work_dir}"/egs/pupuvocoder/exp_config_pupuvocoder.json \
    --infer_datasets $infer_datasets \
    --vocoder_dir $infer_expt_dir \
    --output_dir $infer_output_dir  \
    --log_level debug