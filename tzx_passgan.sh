passgan sample \
 --input-dir ./pretrained \
 --checkpoint ./pretrained/checkpoints/checkpoint_5000.ckpt \
 --output gen_passwords.txt \
 --batch-size 1024 \
 --num-samples 100000
