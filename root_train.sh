 python train.py \
 --data-train 'dataset/train_root/ntuple_merged_*.root' \
 --data-config data/ak15_points_pf_sv.yaml \
 --network-config networks/in_pf_sv.py  \
 --model-prefix model_checkpoints/root10-29 \
 --num-workers 5 \
 --gpus 0,1,2,3 \
 --batch-size 512 \
 --start-lr 5e-3 \
 --num-epochs 20 \
 --optimizer ranger  | tee logs/train.log
