# Sys-NL-Feedback
This is the code for the EACL 2024 paper [System-Level Natural Language Feedback](https://arxiv.org/abs/2306.13588)

## Requirements
Our code is mostly based on [ParlAI](https://parl.ai/#getstarted), we only modified `ParlAI/parlai/core/torch_agent.py` and `ParlAI/parlai/tasks/fits/agents.py`. To install all the requirements, run the following
```bash
cd ParlAI
python setup.py develop
```
## Data
Download and uncompress the data file from [here](https://drive.google.com/file/d/1kfL8x3lq16d_4-Vv2Nm1KLlDaCh1NkMu/view?usp=sharing). Data for training the query generator and response generator are in the `data` folder.


## Model
### Query Generator
To train the query generator, modify the last lines in `ParlAI/parlai/tasks/fits/agents.py` into the following
```python
class DefaultTeacher(MyQueryTeacher):
    pass
```
Then, run the following for training. Please replace the placeholders `<XX>` with parameters you intend to use.
```bash
parlai train_model --task fits --model-file <MODEL_NAME>/model \
--init-model zoo:blenderbot2/query_generator/model --model bart --batchsize <BATCH_SIZE> --num-epochs <NUM_EPOCHS> \
--beam-min-length 2 --inference beam --fp16-impl mem_efficient --optimizer mem_eff_adam \
--learningrate 7e-06 --truncate 512 --text-truncate 512 --label-truncate 128 \
--my-train-data data/query_satisfied_unsatisfied.txt
```

You can evaluate the trained query generator using the following commands. Please replace the placeholders `<XX>` with parameters you intend to use.
```bash
# valid
parlai eval_model --task fits --model-file <MODEL_NAME>/model --datatype valid --display-examples False
# test
parlai eval_model --task fits --model-file <MODEL_NAME>/model --datatype test --display-examples False
# test unseen
parlai eval_model --task fits --model-file <MODEL_NAME>/model --datatype test --unseen-task True --display-examples False
```

### Response Generator
To train the response generator, modify the last lines in `ParlAI/parlai/tasks/fits/agents.py` into the following
```python
class DefaultTeacher(MyFitsBaseTeacher):
    pass
```
Then run the following to train the response generator with satisfied data and unsatisfied data that are refined using system-level feedback only (can change the `--my-train-data` with other data as well). Please replace the placeholders `<XX>` with parameters you intend to use.
```bash
parlai train_model --task fits \
--model-file <RESPONSE_GENERATOR>/model \
--init-model zoo:blenderbot2/blenderbot2_400M/model --model projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent \
--dict-file zoo:blenderbot2/blenderbot2_400M/model.dict --datatype train --batchsize <BATCH_SIZE> --update-freq <UPDATE_FREQUENCY> \
--embedding-size 1024 \
--ffn-size 4096 --dropout 0.1 --attention-dropout 0.1 --n-heads 16 --learn-positional-embeddings True \
--embeddings-scale True --n-positions 1024 --variant prelayernorm --activation gelu --n-encoder-layers 12 \
--n-decoder-layers 12 --model-parallel True --generation-model bart --query-model bert_from_parlai_rag \
--rag-retriever-type dpr --max-doc-token-length 256 --dpr-model-file zoo:hallucination/bart_rag_token/model \
--beam-size 10 --beam-min-length 20 --beam-context-block-ngram 3 --beam-block-ngram 3 --beam-block-full-context False \
--inference beam --fp16 False --fp16-impl safe --optimizer adam --learningrate 7e-6 --warmup-updates 0 \
--truncate 512 --text-truncate 512 --label-truncate 128 --gradient-clip 0.1 \
--lr-scheduler reduceonplateau --lr-scheduler-patience 3 \
--dict-tokenizer gpt2 \
--query-generator-model-file <QUERY_GENERATOR>/model --search-query-generator-beam-min-length 2 \
--knowledge-access-method all --splitted-chunk-length 1000 --retriever-debug-index compressed \
--memory-key full_text --insert-gold-docs true --n-docs 5 \
--gold-document-key __retrieved-docs__ --gold-sentence-key __selected-sentences__ --gold-document-titles-key __retrieved-docs-titles__ \
--memory-decoder-model-file '' --max-train-steps <MAX_TRAIN_STEPS> --log-every-n-steps 20 \
--my-train-data data/response_satisfied_unsatisfied.txt
```
You can evaluate the trained response generator using the following commands. Please replace the placeholders `<XX>` with parameters you intend to use.
```bash
# Valid
parlai eval_model --model-file <RESPONSE_GENERATOR>/model \
--query-generator-model-file <QUERY_GENERATOR>/model \
--task fits --datatype valid \
--splitted-chunk-length 1000 --retriever-debug-index compressed \
--memory-key full_text --insert-gold-docs true --n-docs 5 \
--gold-document-key __retrieved-docs__ --gold-sentence-key __selected-sentences__ \
--gold-document-titles-key __retrieved-docs-titles__ --batchsize 1 \
--memory-decoder-model-file '' --rag-retriever-type dpr \
--my-valid-data data/eval_responses/valid.json --display-examples False

# Test
parlai eval_model --model-file <RESPONSE_GENERATOR>/model \
--query-generator-model-file <QUERY_GENERATOR>/model \
--task fits --datatype test \
--splitted-chunk-length 1000 --retriever-debug-index compressed \
--memory-key full_text --insert-gold-docs true --n-docs 5 \
--gold-document-key __retrieved-docs__ --gold-sentence-key __selected-sentences__ \
--gold-document-titles-key __retrieved-docs-titles__ --batchsize 1 \
--memory-decoder-model-file '' --rag-retriever-type dpr \
--my-test-data data/eval_responses/test.json --display-examples False

# Test Unseen
parlai eval_model --model-file <RESPONSE_GENERATOR>/model \
--query-generator-model-file <QUERY_GENERATOR>/model \
--task fits --datatype test --unseen-task True \
--splitted-chunk-length 1000 --retriever-debug-index compressed \
--memory-key full_text --insert-gold-docs true --n-docs 5 \
--gold-document-key __retrieved-docs__ --gold-sentence-key __selected-sentences__ \
--gold-document-titles-key __retrieved-docs-titles__ --batchsize 1 \
--memory-decoder-model-file '' --rag-retriever-type dpr \
--my-test-unseen-data data/eval_responses/test_unseen.json --display-examples False
```
## Bib
Please cite our work if you find it useful.
```
@inproceedings{yuan-etal-2024-system,
    title = "System-Level Natural Language Feedback",
    author = "Yuan, Weizhe  and
      Cho, Kyunghyun  and
      Weston, Jason",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.169",
    pages = "2773--2789",
    abstract = "Natural language (NL) feedback offers rich insights into user experience. While existing studies focus on an instance-level approach, where feedback is used to refine specific examples, we introduce a framework for system-level use of NL feedback. We show how to use feedback to formalize system-level design decisions in a human-in-the-loop-process {--} in order to produce better models. In particular this is done through: (i) metric design for tasks; and (ii) language model prompt design for refining model responses. We conduct two case studies of this approach for improving search query and dialog response generation, demonstrating the effectiveness of system-level feedback. We show the combination of system-level and instance-level feedback brings further gains, and that human written instance-level feedback results in more grounded refinements than GPT-3.5 written ones, underlying the importance of human feedback for building systems.",
}
```

