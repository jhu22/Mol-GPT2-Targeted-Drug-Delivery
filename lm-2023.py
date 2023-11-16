import smiles_gpt as gpt
filename = "data/temp2023.txt"
checkpoint = "checkpoints/temp2023"

hyperparams = {"batch_size": 256, "max_epochs": 30, "min_epochs": 15,
               "max_length": 512, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 12, "n_embd": 12 * 48}

num_workers = 32  # Number of dataloader worker processes.
is_tokenizer_pretrained = True

tokenizer = gpt.SMILESBPETokenizer(dropout=None)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = tokenizer.from_file('checkpoints/benchmark-10m/vocab.json','checkpoints/benchmark-10m/merges.txt')

from pprint import pprint

tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer('checkpoints/benchmark-10m/tokenizer.json', model_max_length=hyperparams["max_length"])

smiles_string = "CC(Cl)=CCCC=C(C)Cl"
smiles_encoded = tokenizer(smiles_string)
smiles_merges = tokenizer.convert_ids_to_tokens(smiles_encoded["input_ids"])

pprint(smiles_encoded)
pprint(smiles_merges)

datamodule = gpt.LMDataModule(filename, tokenizer,
                              batch_size=hyperparams["batch_size"],
                              num_workers=num_workers)

from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(vocab_size=tokenizer.vocab_size,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    n_layer=hyperparams["n_layer"],
                    n_head=hyperparams["n_head"],
                    n_embd=hyperparams["n_embd"],
                    n_positions=hyperparams["max_length"],
                    n_ctx=hyperparams["max_length"])
model = GPT2LMHeadModel(config)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

checkpoint_cb = ModelCheckpoint(f"{checkpoint}/model/")

early_stopping_ppl = EarlyStopping(
    monitor="ppl_epoch",
    patience=4,
    min_delta=5e-3,
    check_finite=True,
    stopping_threshold=1.1,
    divergence_threshold=hyperparams["vocab_size"] / 10,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=True,
)

trainer = Trainer(
    strategy="ddp",
    callbacks=[checkpoint_cb, early_stopping_ppl],
    max_epochs=hyperparams["max_epochs"],
    min_epochs=hyperparams["min_epochs"],
    val_check_interval=0.4,
    limit_train_batches=0.5,
    log_every_n_steps=200,
)
lit_model = gpt.GPT2LitModel(
    model,
    batch_size=hyperparams["batch_size"],
    learning_rate=hyperparams["learning_rate"],
    final_learning_rate=hyperparams["final_learning_rate"],
    weight_decay=hyperparams["weight_decay"],
    adam_eps=hyperparams["adam_eps"],
    adam_betas=hyperparams["adam_betas"],
    scheduler_T_max=hyperparams["scheduler_T_max"],
)
trainer.fit(lit_model, datamodule)

lit_model.transformer.save_pretrained(f"{checkpoint}/model/")

