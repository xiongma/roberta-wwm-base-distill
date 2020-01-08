# A Roberta-ext-wwm-base Distillation Model

This is a chinese Roberta wwm distillation model which was distilled from roberta-ext-wwm-large by roberta-ext-wwm-large. The large model and base model are from this [github]([https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)), thanks for his contribution.

## Based On

This model was trained based on this [paper]([https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)), which was pushed by huggingfaace.

## Corpus

For train this model, I used baike_qa2019, news2016_zh,  webtext_2019, wiki_zh. this data can be found in this [github]([https://github.com/brightmart/nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus))

## Model Download

I just BaiduYun to down this model, this link is below.

| Model                                 | BaiduYun                                                      |
|:-------------------------------------:|:-------------------------------------------------------------:|
| Roberta-wwm-ext-base-distill, Chinese | [Tensorflow](https://pan.baidu.com/s/1R3f1nREQ4qKiloiZn2gFbQ) |

## Train Detail

### Dataset

- I just used 5 different ways to mask one sentence, not dynamic mask.

- Every example just use maximum 20 tokens mask

### Teacher Model

- I used Roberta large model to get every masked token's output, which was mapped to vocab, I just kept max 128 dimensions, you could ask why didn't you keep more dimensions, first, the storge is too much, second, I think keep too much is unneccessary.

### Student Model

- **Loss**: In this training, I use 2 loss functions, first is cross entropy, second is cosin loss, add them together, I think it has a big improvement if I use another loss function, but I didn't have too much resource to train this model, because my free Google TPU expired.

- **Other parameters**

| parameter     | Value |
|:-------------:|:-----:|
| batch size    | 384   |
| learning rate | 5e-5  |
| training step | 1M    |
| warming step  | 2W    |

## Comprasion

In this part, every task I just ran one time, the result is below.

### Classification

| Model                                 | AFQMC      | CMNLI     | TNEWS     |
|:-------------------------------------:|:----------:|:---------:|:---------:|
| Roberta-wwm-ext-base, Chinese         | 74.04%     | 80.51%    | 56.94%    |
| Roberta-wwm-ext-base-distill, Chinese | **74.44%** | **81.1%** | **57.6%** |

### SQUAD

| Model                                 | CMRC dev    |
|:-------------------------------------:|:-----------:|
| Roberta-wwm-ext-base, Chinese         | 84.72/65.24 |
| Roberta-wwm-ext-base-distill, Chinese | 85.2        |

In this part you could ask, your comprasion is different with this [github]([https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)), I don't why, I just used the original base model to run this task, got the score is up, and I used same parameters and distilled model to run this task, got the score is up. Maybe I used the different parameters, but as you can see,  in the same situation, the distilled model has improvement.


