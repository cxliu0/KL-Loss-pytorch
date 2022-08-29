CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    ./configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco_kl.py \
    /path/to/model_checkpoint \
     --eval 'bbox'