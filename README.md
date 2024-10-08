# Synthesizing interpretable control policies

This repo contains the code associated to the paper "Synthesizing Interpretable Control Policies through Large Language Model Guided Search".

## Reference

If you use this code in an academic context, please cite the publication:

```

```

## Usage

In our implementation, the LLM we use is [Starcoder2](https://github.com/bigcode-project/starcoder2) through the [Ollama](https://ollama.com/) API. We run the model locally on a RTX3090 GPU. As an alternative, the OpenAI APIs can also be used. We define some model hyperparameters in the `Modelfile`.
Once Ollama is installed, to instantiate the model, run:
```
ollama create starcoder2:control -f Modelfile
```
We run the algorithm on Docker. The implementation is taken from [this](https://github.com/jonppe/funsearch) repo, which is itself a fork from the DeepMind [FunSearch](https://github.com/google-deepmind/funsearch) repo.

You can run FunSearch in container using Docker. There are variations to how to make the LLM interface with the container. These are the commands that we used:

```
docker build . -t funsearch

# Create a folder to share with the container
mkdir data

docker run --network host -it -v /home/cbosio/fun-design/data:/workspace/data funsearch

# [carlo] to run dm_control (the number is #episodes)
funsearch run examples/dm_control_swingup_spec.py 1 --sandbox_type ExternalProcessSandbox
funsearch run examples/dm_control_ballcup_spec.py 1 --sandbox_type ExternalProcessSandbox
```

You should see output something like

```
INFO:root:Writing logs to data/1704956206
INFO:absl:Best score of island 0 increased to 2048
INFO:absl:Best score of island 1 increased to 2048
INFO:absl:Best score of island 2 increased to 2048
INFO:absl:Best score of island 3 increased to 2048
INFO:absl:Best score of island 4 increased to 2048
INFO:absl:Best score of island 5 increased to 2048
INFO:absl:Best score of island 6 increased to 2048
INFO:absl:Best score of island 7 increased to 2048
INFO:absl:Best score of island 8 increased to 2048
INFO:absl:Best score of island 9 increased to 2048
INFO:absl:Best score of island 5 increased to 2053
INFO:absl:Best score of island 1 increased to 2049
INFO:absl:Best score of island 8 increased to 2684
^C^CINFO:root:Keyboard interrupt. Stopping.
INFO:absl:Saving backup to data/backups/program_db_priority_1704956206_0.pickle.
```

## Tests
If you are interested in seeing the performances of the policies presented in the paper, just run the scripts in the `/dm_control_tests` folder.

For more implementation details, check [this](https://github.com/jonppe/funsearch) repo. The original research work can be found at

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)

## Contact
Please contact c.bosio@berkeley.edu if you have questions.