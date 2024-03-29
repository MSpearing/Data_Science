python compressed_sensing.py \
    --dataset=celebA \
    --input-type=full-input \
    --num-input-images=1 \
    --model-types dcgan-discrim \
    --num-measurements=5000 \
    --noise-std=0.1 \
    --optimizer-type=momentum \
    --learning-rate=0.1 \
    --momentum=0.9 \
    --max-update-iter=100 \
    --num-random-restarts=1 \
    --lmbd 0.1 \
    --print-stats \
    --save-images \
    --image-matrix=3
