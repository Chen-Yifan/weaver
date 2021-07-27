 ip=$(hostname -i)
 
 python train.py \
 --data-train 'dataset/train_root/ntuple_merged_*.root' \
 --data-config data/ak15_points_pf_sv.yaml \
 --network-config networks/in_pf_sv.py  \
 --model-prefix model_checkpoints/all_data \
 --num-workers 2 \
 --gpus 2 \
 --ipaddr $ip \
 --batch-size 512 \
 --start-lr 5e-3 \
 --num-epochs 20 \
 --optimizer ranger  | tee logs/train.log
