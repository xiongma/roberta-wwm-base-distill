# A Roberta-ext-wwm-base Distillation Model

This is a chinese Roberta wwm distillation model which was distilled from roberta-ext-wwm-base by roberta-ext-wwm-large. The large model and base model are from this [github](https://github.com/ymcui/Chinese-BERT-wwm), thanks for his contribution.

## Based On

This model was trained based on this [paper](https://arxiv.org/abs/1910.01108), which was punished by huggingface.

## Corpus

For train this model, I used baike_qa2019, news2016_zh,  webtext_2019, wiki_zh. this data can be found in this [github](https://github.com/brightmart/nlp_chinese_corpus)

## Model Download

I just support BaiduYun to down this model, this link is below.

| Model                                    | BaiduYun                                                      |
|:----------------------------------------:|:-------------------------------------------------------------:|
| Roberta-wwm-ext-base-distill, Chinese    | [Tensorflow](https://pan.baidu.com/s/1e7_Zqa1_TlFfzx1n51BTUw) |
| Roberta-wwm-ext-3layers-distill, Chinese | Tensorflow                                                    |

## Train Detail

To train this model, I used 2 steps.

- I used roberta_ext_wwm_large model to get all examples tokens' output.

- I used the output to train the model, which inited roberta_ext_wwm_base pretrain model weights.

### Dataset

- I just used 5 different ways to mask one sentence, not dynamic mask.

- Every example just use maximum 20 token masks

### Teacher Model

- I used Roberta large model to get every masked token's output, which was mapped to vocab, I just kept max 128 dimensions, you could ask why didn't you keep more dimensions, first, the storge is too much, second, I think keep too much is unneccessary.

### Student Model

- **Loss**: In this training, I use 2 loss functions, first is cross entropy, second is cosin loss, add them together, I think it has a big improvement if I use another loss function, but I didn't have too much resource to train this model, because my free Google TPU expired.

- **Other Parameters**

| Parameter     | Value |
|:-------------:|:-----:|
| batch size    | 384   |
| learning rate | 5e-5  |
| training step | 1M    |
| warming step  | 2W    |

## Comparison

In this part, every task I just ran one time, the result is below.

### Classification

| Model                                    | AFQMC      | CMNLI     | TNEWS     |
|:----------------------------------------:|:----------:|:---------:|:---------:|
| Roberta-wwm-ext-base, Chinese            | 74.04%     | 80.51%    | 56.94%    |
| Roberta-wwm-ext-base-distill, Chinese    | **74.44%** | **81.1%** | **57.6%** |
| Roberta-wwm-ext-3layers-distill, Chinese | 68.8%      | 75.5%     | 55.7%     |

| Model                                    | LCQMC dev | LCQMC test |
|:----------------------------------------:|:---------:|:----------:|
| Roberta-wwm-ext-base, Chinese            | 89%       | 86.5%      |
| Roberta-wwm-ext-base-distill, Chinese    | 89%       | **87.2%**  |
| Roberta-wwm-ext-3layers-distill, Chinese | 85.1%     | 86%        |

### SQUAD

| Model                                    | CMRC2018 dev (F1/EM) |
|:----------------------------------------:|:--------------------:|
| Roberta-wwm-ext-base, Chinese            | 84.72%/**65.24%**    |
| Roberta-wwm-ext-base-distill, Chinese    | **85.2%**/65.20%     |
| Roberta-wwm-ext-3layers-distill, Chinese | 78.5%/57.4%          |

In this part you could ask, your comparison is different with this [github](https://github.com/ymcui/Chinese-BERT-wwm), I don't know why, I just used the original base model to run this task, got the score is up, and I used same parameters and distilled model to run this task, got the score is up. Maybe I used the different parameters. 

But as you can see,  in the same situation, the distilled model has improvement than the original model.

## How To Train

- **create pretraining data**

```python
export DATA_DIR=YOUR_DATA_DIR
export OUTPUT_DIR=YOUR_OUTPUT_DIR
export VOCAB_FILE=YOUR_VOCAB_FILE

python create_pretraining_data.py \
        --input_dir=$DATA_DIR\
        --output_dir=$OUTPUT_DIR \
        --vocab_file=$YOUR_VOCAB_FILE \
        --do_whole_word_mask=True \
        --ramdom_next=True \
        --max_seq_length=512 \
        --max_predictions_per_seq=20 \
        --random_seed=12345 \
        --dupe_factor=5 \
        --masked_lm_prob=0.15 \
        --doc_stride=256 \
        --max_workers=2 \
        --short_seq_prob=0.1
```

- **create teacher output data**

```python
export TF_RECORDS=YOUR_PRETRAINING_TF_RECORDS
export TEACHER_MODEL=YOUR_TEACHER_MODEL_DIR
export OUTPUT_DIR=YOUR_OUTPUT_DIR

python create_teacher_output_data.py \
       --bert_config_file=$TEACHER_MODEL/bert_config.json \
       --input_file=$TF_RECORDS \
       --output_dir=$YOUR_OUTPUT_DIR \
       --truncation_factor=128 \
       --init_checkpoint=$TEACHER_MODEL\bert_model.ckpt \
       --max_seq_length=512 \
       --max_predictions_per_seq=20 \
       --predict_batch_size=64 
```

- **run distill**

```python
export TF_RECORDS=YOUR_TEACHER_OUTPUT_TF_RECORDS
export STUDENT_MODEL_DIR=YOUR_STUDENT_MODEL_DIR
export OUTPUT_DIR=YOUR_OUTPUT_DIR

python run_distill.py \
       --bert_config_file=$STUDENT_MODEL_DIR\bert_config.json \
       --input_file=$TF_RECORDS \
       --output_dir=$OUTPUT_DIR \
       --init_checkpoint=$STUDENT_MODEL_DIR\bert_model.ckpt
       --truncation_factor=128 \
       --max_seq_length=512 \
       --max_predictions_per_seq=20 \
       --do_train=True \
       --do_eval=True \
       --train_batch_size=384 \
       --eval_batch_size=1024 \
       --num_train_steps=1000000 \
       --num_warmup_steps=20000 
```

## Answers

- **We need a small size one, your model are still base size.**
1. The purpose of punish this model is to identify feasibility of distilled of method.

2. As you can see, this distilled method can improve the accuracy.
   
   
- **Why did you punish the 3 layers model?**
1. Some githuber told me, we need small size one, the bert base version is so large, I can't afford the cost of the server, so I punished the small size one! 



- **Future Plan**

- I still trained a 6 layers model, **I will punish it around 2020.01.23**

## Thanks

Thanks TFRC supports the TPU!
