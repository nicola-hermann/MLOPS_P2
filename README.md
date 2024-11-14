# MLOPS_P2

This repository is for the HSLU HS24 MLOPS course. This is the 2nd Project. It is all about containerization. 

## Run with local Docker

If you have Docker installed on your system, you can clone this repository and open it in the root directory.

For ease of use, I recommend creating a .env file at the root directory level of this repository. In there, you specify your WandB API key: `WANDB_API_KEY=<your_api_key>`

You are now all set to build and run the docker image by typing:
```
docker build -t mlopsp2 .

docker run --env-file .env mlopsp2
```

In this example, I have chosen mlopsp2 as the image name. This docker run command runs the python script with default arguments. To run it with different arguments just add pythun run.py and the arguments at the end of the run command eg.
```
docker run --env-file .env mlopsp2 python run.py --warmup_steps 20 
```

Here is a list of all the possible arguments, that you can give to the python file:
- epochs
- learning_rate
- optimizer (Options: AdamW or SGD)
- warmup_steps
- weight_decay
- lr_scheduler (Options: linear, cosine, constant)
- train_batch_size
- eval_batch_size
- save_path
- seed
- wandb_project
- wandb_entity (Team or Username)

## Run with GitHub Codespaces

In order to run it with GitHub Codespaces, the previous steps apply except for the .env file.

We skip the .env file and pass the WandB API key by running the command:
```
docker run -e WANDB_API_KEY=<your_api_key> <image-name>
```

Everything else also applies for GitHub Codespaces.

