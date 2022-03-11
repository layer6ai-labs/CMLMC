#!/usr/bin/env bash
python train.py "data-bin/iwslt14_deen_jointdict" \
    --arch cmlm_transformer_iwslt_en_de \
    -s de \
    -t en \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --criterion nat_loss \
    --task translation_lev \
    --label-smoothing 0.1 \
    --noise random_mask \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' \
    --lr 0.0005 \
    --warmup-updates 10000 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --share-all-embeddings \
    --max-tokens 8192 \
    --max-epoch 300 \
    --fixed-validation-seed 7 \
    --fp16 \
    --save-dir ./results/checkpoints/IWSLTdeen_raw_CMLM_benchmark/ \

python InferenceIWSLT_valid.py IWSLTdeen_raw_CMLM_benchmark 80 300

python train.py "data-bin/iwslt14_deen_jointdict" \
   --arch cmlm_transformer_iwslt_en_de \
   -s de \
   -t en \
   --optimizer adam \
   --adam-betas '(0.9,0.98)' \
   --criterion nat_loss \
   --task translation_lev \
   --label-smoothing 0.1 \
   --noise random_mask \
   --lr-scheduler inverse_sqrt \
   --warmup-init-lr '1e-07' \
   --lr 0.0005 \
   --warmup-updates 30000 \
   --dropout 0.3 \
   --weight-decay 0.01 \
   --decoder-learned-pos \
   --encoder-learned-pos \
   --apply-bert-init \
   --share-all-embeddings \
   --max-tokens 8192 \
   --max-epoch 300 \
   --fixed-validation-seed 7 \
   --fp16 \
   --no-scale-embedding \
   --insertCausalSelfAttn \
   --concatPE \
   --selfcorrection 0 \
   --replacefactor 0.3 \
   --save-dir ./results/checkpoints/IWSLTdeen_raw_CMLMC_L5D3_30k/ \

python InferenceIWSLT_valid.py IWSLTdeen_raw_CMLMC_L5D3_30k 150 300


