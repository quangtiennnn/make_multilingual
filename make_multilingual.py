import logging
from datetime import datetime
import numpy as np
from datasets import DatasetDict, load_dataset
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    MSEEvaluator,
    SequentialEvaluator,
    TranslationEvaluator,
)
from sentence_transformers.losses import MSELoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

# Model and training parameters
teacher_model_name = "paraphrase-distilroberta-base-v2"
student_model_name = "vinai/phobert-base-v2"

student_max_seq_length = 128
train_batch_size = 64
inference_batch_size = 64
max_sentences_per_language = 500000
num_train_epochs = 5
num_evaluation_steps = 5000

# Languages
source_languages = set(["en"])
target_languages = set(["vi"])

# Output directory
output_dir = (
    "output/make-multilingual-"
    + "-".join(sorted(list(source_languages)) + sorted(list(target_languages)))
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Define teacher and student models
teacher_model = SentenceTransformer(teacher_model_name)
student_model = SentenceTransformer(student_model_name)
student_model.max_seq_length = student_max_seq_length
logging.info(f"Teacher model: {teacher_model}")
logging.info(f"Student model: {student_model}")

# Load parallel sentences dataset
dataset_to_use = "harouzie/vi_en-translation"
train_dataset_dict = DatasetDict()
eval_dataset_dict = DatasetDict()
for target_lang in target_languages:
    subset = f"en-{target_lang}"
    try:
        train_dataset = load_dataset(dataset_to_use, subset, split="train")
        if len(train_dataset) > max_sentences_per_language:
            train_dataset = train_dataset.select(range(max_sentences_per_language))
    except Exception as exc:
        logging.error(f"Could not load dataset {dataset_to_use}/{subset}: {exc}")
        continue

    try:
        eval_dataset = load_dataset(dataset_to_use, subset, split="dev")
        if len(eval_dataset) > 1000:
            eval_dataset = eval_dataset.select(range(1000))
    except Exception:
        logging.info(
            f"Could not load dataset {dataset_to_use}/{subset} dev split, splitting 1k samples from train"
        )
        dataset = train_dataset.train_test_split(test_size=1000, shuffle=True)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    train_dataset_dict[subset] = train_dataset
    eval_dataset_dict[subset] = eval_dataset
logging.info(train_dataset_dict)

# Prepare the dataset
def prepare_dataset(batch):
    return {
        "english": batch["english"],
        "non_english": batch["non_english"],
        "label": teacher_model.encode(batch["english"], batch_size=inference_batch_size, show_progress_bar=False),
    }

column_names = list(train_dataset_dict.values())[0].column_names
train_dataset_dict = train_dataset_dict.map(
    prepare_dataset, batched=True, batch_size=30000, remove_columns=column_names
)
logging.info("Prepared datasets for training:", train_dataset_dict)

# Define the training loss
train_loss = MSELoss(model=student_model)

# Define evaluators
evaluators = []
for subset, eval_dataset in eval_dataset_dict.items():
    logger.info(f"Creating evaluators for {subset}")

    dev_mse = MSEEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name=subset,
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_mse)

    dev_trans_acc = TranslationEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name=subset,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_trans_acc)

    test_dataset = None
    try:
        test_dataset = load_dataset("mteb/sts17-crosslingual-sts", subset, split="test")
    except Exception:
        try:
            test_dataset = load_dataset("mteb/sts17-crosslingual-sts", f"{subset[3:]}-{subset[:2]}", split="test")
            subset = f"{subset[3:]}-{subset[:2]}"
        except Exception:
            pass
    if test_dataset:
        test_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=[score / 5.0 for score in test_dataset["score"]],
            batch_size=inference_batch_size,
            name=f"sts17-{subset}-test",
            show_progress_bar=False,
        )
        evaluators.append(test_evaluator)

evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: np.mean(scores))
eval_dataset_dict = eval_dataset_dict.map(prepare_dataset, batched=True, batch_size=30000, remove_columns=column_names)

# Define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=num_evaluation_steps,
    save_strategy="steps",
    save_steps=num_evaluation_steps,
    save_total_limit=2,
    logging_steps=100,
    run_name=f"multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}",
)

# Create the trainer and start training
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset_dict,
    eval_dataset=eval_dataset_dict,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()

# Save the trained model
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)

# Optional: push the model to the Hugging Face Hub
model_name = student_model_name if "/" not in student_model_name else student_model_name.split("/")[-1]
try:
    student_model.push_to_hub(f"{model_name}-multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}')`."
    )
