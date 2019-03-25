#python3 train_net.py --config-file="../configs/tuo/e2e_faster_rcnn_R_50_C4_1x_1_gpu_voc_roi_7.yaml" OUTPUT_DIR ./checkpoint/roi_7
python3 train_net.py --config-file="../configs/tuo/e2e_faster_rcnn_R_50_FPN_1x.yaml" OUTPUT_DIR ./checkpoint/fpn
python3 train_net.py --config-file="../configs/tuo/e2e_faster_rcnn_R_50_C4_1x_1_gpu_voc.yaml" OUTPUT_DIR ./checkpoint/c4
