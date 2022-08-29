CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch  \
	--nproc_per_node=4  \
	--master_port=11500 \
    ./tools/train.py \
    ./configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco_kl.py \
    --work-dir='./outputs/coco/fasterrcnn_COCO_KL-Loss-pytorch' \
    --launcher pytorch