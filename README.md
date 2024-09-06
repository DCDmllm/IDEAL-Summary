# IDEAL_Summary
**IDEAL: Leveraging Infinite and Dynamic Characterizations of Large Language Models for Query-focused Summarization**

## Data Processing

1. Download datasets from their respective official repositories:

    * [SQuALITY](https://github.com/nyu-mll/SQuALITY/tree/main/data/v1-3/txt)
    * [CovidET](https://github.com/honglizhan/CovidET/tree/main)
    * [QMSum](https://github.com/Yale-LILY/QMSum)

2. Preprocess the datasets using the provided Jupyter notebook: **`data_process.ipynb`**.

## Training, Inference, and Evaluation

To train, run inference, and evaluate the model, execute the following script:

```
bash exps/finetuning_*_generate_evaluate.sh
```

For multi-reference Rouge scores and Bart-score evaluations on the SQuALITY dataset, use the notebook **`multi_reference_evaluation_SQuAlITY.ipynb`**.

---