# DHIM

The Pytorch implementation of paper "Refining BERT Embeddings for Document Hashing via Mutual Information Maximization" (EMNLP 2021).

### Main Dependencies

- torch 1.2.0
- transformers 3.2.0

### Pre-trained model Download

We implement our model in [HugginFace AIP](https://huggingface.co/). Please download the BERT-based model from [this link](https://huggingface.co/bert-base-uncased/tree/main) and put it at `./model/bert-base-uncased/`.

### How to Run

```
# Run with the DBpedia dataset
python main.py dbpedia32 ./data/dbpedia --train --seed 32236 --batch_size 512 --epochs 100 --lr 0.001 --encode_length 32 --cuda --max_length 50 --distance_metric hamming --num_retrieve 100 --num_bad_epochs 6 --clip 10.0 --alpha 0.1 --beta 0.4 --conv_out_dim 256
```

To reproduce the results reported in the paper, please refer to the `run.sh` for detailed running comments.