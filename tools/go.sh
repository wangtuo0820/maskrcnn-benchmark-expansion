#python3 train_net.py --config-file="../configs/tuo/e2e_faster_rcnn_R_50_C4_1x_1_gpu_voc_roi_7.yaml" OUTPUT_DIR ./checkpoint/roi_7
#python3 train_net.py --config-file="../configs/tuo/e2e_faster_rcnn_R_34_FPN_1x.yaml" OUTPUT_DIR ./checkpoint/res34_fpn
python3 train_net.py --config-file="../configs/tuo/e2e_faster_rcnn_R_18_C4_1x_1_gpu_voc.yaml" OUTPUT_DIR ./checkpoint/res18
#python3 train_net.py --config-file="../configs/tuo/e2e_faster_rcnn_R_50_C4_1x_1_gpu_voc.yaml" OUTPUT_DIR ./checkpoint/res50
