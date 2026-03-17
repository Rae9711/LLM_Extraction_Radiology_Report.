import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import (
	AutoConfig,
	AutoTokenizer,
	AutoModel,
	PreTrainedModel,
	get_linear_schedule_with_warmup,
)
import torchtuples as tt 
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from tqdm import tqdm


def _breslow_baseline_cumulative(durations: np.ndarray, events: np.ndarray, exp_g: np.ndarray) -> pd.Series:
	"""Breslow cumulative baseline hazard at event times."""
	order = np.argsort(durations)
	d, e, eg = durations[order], events[order], exp_g[order]
	event_times = np.unique(d[e != 0])
	if len(event_times) == 0:
		return pd.Series([0.0], index=[0.0])
	at_risk_sum = np.array([np.sum(eg[d >= t]) for t in event_times])
	event_counts = np.array([np.sum((d == t) & (e != 0)) for t in event_times])
	cum = np.cumsum(event_counts / np.maximum(at_risk_sum, 1e-12))
	return pd.Series(cum, index=event_times)


def _get_g_from_loader(loader: DataLoader, model: nn.Module, device: torch.device) -> tuple:
	"""Return (durations, events, g) from loader; g = model output (log hazard)."""
	model.eval()
	durations_list, events_list, g_list = [], [], []
	with torch.no_grad():
		for x_batch, y in loader:
			x_batch = {k: v.to(device) for k, v in x_batch.items()}
			durations, events = y[0], y[1]
			durations_list.append(durations.numpy())
			events_list.append(events.numpy())
			g = model(**x_batch)["output"].squeeze(-1).cpu().numpy()
			g_list.append(g)
	model.train()
	return np.concatenate(durations_list), np.concatenate(events_list), np.concatenate(g_list)


def _coxph_surv_df(
	g_eval: np.ndarray,
	baseline_cumulative: pd.Series,
	time_grid: np.ndarray,
) -> pd.DataFrame:
	"""S(t|x) = exp(-H_0(t) * exp(g(x)))."""
	idx = np.searchsorted(baseline_cumulative.index.values, time_grid, side="right") - 1
	idx = np.maximum(idx, 0)
	H0 = baseline_cumulative.iloc[idx].values
	exp_g = np.exp(g_eval).reshape(1, -1)
	cumhaz = H0.reshape(-1, 1) * exp_g
	return pd.DataFrame(np.exp(-cumhaz), index=time_grid)


def _integrated_brier_manual(ev: EvalSurv, time_grid: np.ndarray) -> float:
	"""Integrated Brier score using np.trapz (avoid pycox's scipy.simps)."""
	scores = ev.brier_score(time_grid)
	scores = np.asarray(scores.values if hasattr(scores, "values") else scores, dtype=np.float64)
	if scores.size == 0 or time_grid[-1] <= time_grid[0]:
		return float(scores[0]) if scores.size else float("nan")
	return float(np.trapz(scores, time_grid) / (time_grid[-1] - time_grid[0]))

class SurvivalTextDataset(Dataset):

	def __init__(
		self,
		df: pd.DataFrame,
		tokenizer: AutoTokenizer,
		max_length: int = 256,
	):
		self.df = df.reset_index(drop=True)
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		row = self.df.iloc[idx]
		text = str(row["text"])

		embedding = self.tokenizer(
			text,
			padding="max_length",
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		)

		item = {
			"input_ids": torch.tensor(embedding["input_ids"], dtype=torch.long),
			"attention_mask": torch.tensor(embedding["attention_mask"], dtype=torch.long),
			"event": torch.tensor(row["event"], dtype=torch.float32),
			"duration": torch.tensor(row["duration"], dtype=torch.float32),
		}
		if "token_type_ids" in embedding:
			item["token_type_ids"] = torch.tensor(
				embedding["token_type_ids"], dtype=torch.long
			)
		x = {
			"input_ids": embedding["input_ids"].squeeze(0),
			"attention_mask": embedding["attention_mask"].squeeze(0),
		}
		y = (
			item['duration'],
			item['event']
		)
		return x,y


class BertForSurvival(PreTrainedModel):

	config_class = AutoConfig

	def __init__(self, config, num_nodes=[32,32], dropout=0.1,batch_norm=True):
		super().__init__(config)

		self.encoder = AutoModel.from_config(config)

		# for param in self.encoder.parameters():
		#     param.requires_grad = False
		
		# for layer in self.encoder.encoder.layer[-2:]:
		#     for param in layer.parameters():
		#         param.requires_grad = True

		hidden_size = config.hidden_size
		
		# Handle different dropout attribute names across model types
		dropout_prob = getattr(config, 'hidden_dropout_prob', 
							  getattr(config, 'dropout', 
									 getattr(config, 'dropout_rate', 0.1)))
		self.dropout = nn.Dropout(dropout_prob)
		in_features = config.hidden_size
		out_features = 1
		self.survival_head = tt.practical.MLPVanilla(
			in_features,
			num_nodes,
			out_features,
			batch_norm,
			dropout
		)
		self.post_init()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		**kwargs,
	) -> Dict[str, torch.Tensor]:
		encoder_inputs = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
		}
		if token_type_ids is not None and hasattr(self.encoder, "embeddings") and hasattr(self.encoder.embeddings, 'token_type_embeddings'):
			encoder_inputs['token_type_ids'] = token_type_ids
		outputs = self.encoder(**encoder_inputs)
		last_hidden_state = outputs.last_hidden_state 
		cls_emb = last_hidden_state[:, 0, :] 

		
		output = self.survival_head(cls_emb)

		return {"output": output}


@dataclass
class TrainingConfig:

	model_name: str = "roberta-base"
	max_length: int = 256
	test_size: float = 0.2
	random_state: int = 42

	num_train_epochs: int = 3
	train_batch_size: int = 32
	eval_batch_size: int = 32
	learning_rate: float = 5e-5
	weight_decay: float = 0.01

	logging_steps: int = 50
	eval_steps: int = 200
	save_steps: int = 200

	output_dir: str = "./outputs_survival"
	data_path: str = ""


def parse_args():
	parser = argparse.ArgumentParser(description="Train survival analysis model on text data")
	
	parser.add_argument(
		"--base_model",
		type=str,
		default="roberta-base",
		help="Pretrained model name or path (e.g., roberta-base, bert-base-uncased)"
	)
	parser.add_argument(
		"--data_path",
		type=str,
		required=True,
		help="Path to the CSV file containing the dataset"
	)
	parser.add_argument(
		"--run_name",
		type=str,
		required=True,
		help="Name for this training run"
	)
	
	# Training hyperparameters
	parser.add_argument(
		"--batch_size",
		type=int,
		default=32,
		help="Training and evaluation batch size"
	)
	parser.add_argument(
		"--max_length",
		type=int,
		default=256,
		help="Maximum sequence length for tokenization"
	)
	parser.add_argument(
		"--num_epochs",
		type=int,
		default=3,
		help="Number of training epochs"
	)
	parser.add_argument(
		"--learning_rate",
		type=float,
		default=5e-5,
		help="Learning rate for optimizer"
	)
	parser.add_argument(
		"--weight_decay",
		type=float,
		default=0.01,
		help="Weight decay for optimizer"
	)
	parser.add_argument(
		"--test_size",
		type=float,
		default=0.2,
		help="Proportion of dataset to use for testing"
	)
	parser.add_argument(
		"--random_state",
		type=int,
		default=42,
		help="Random seed for reproducibility"
	)
	parser.add_argument(
		"--logging_steps",
		type=int,
		default=50,
		help="Log metrics every N steps"
	)
	parser.add_argument(
		"--eval_steps",
		type=int,
		default=200,
		help="Evaluate model every N steps"
	)
	parser.add_argument(
		"--save_steps",
		type=int,
		default=200,
		help="Save checkpoint every N steps"
	)
	
	return parser.parse_args()


def main():
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	output_dir = os.path.join("./runs", args.run_name)
	os.makedirs(output_dir, exist_ok=True)
	
	cfg = TrainingConfig(
		model_name=args.base_model,
		max_length=args.max_length,
		test_size=args.test_size,
		random_state=args.random_state,
		num_train_epochs=args.num_epochs,
		train_batch_size=args.batch_size,
		eval_batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		logging_steps=args.logging_steps,
		eval_steps=args.eval_steps,
		save_steps=args.save_steps,
		output_dir=output_dir,
		data_path=args.data_path,
	)
	
	print("="*60)
	print(f"Training Configuration for run: {args.run_name}")
	print("="*60)
	print(f"Model: {cfg.model_name}")
	print(f"Data: {cfg.data_path}")
	print(f"Output Directory: {cfg.output_dir}")
	print(f"Batch Size: {cfg.train_batch_size}")
	print(f"Learning Rate: {cfg.learning_rate}")
	print(f"Epochs: {cfg.num_train_epochs}")
	print(f"Max Length: {cfg.max_length}")
	print("="*60 + "\n")

	tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

	print(f"Loading data from {cfg.data_path}...")
	df = pd.read_csv(cfg.data_path)
	print(f"Total samples: {len(df)}")

	# train_df = df.iloc[:7024]
	# eval_df = df.iloc[7024:]
	
	train_df, eval_df = train_test_split(
		df,
		test_size=cfg.test_size,
		random_state=cfg.random_state,
		shuffle=True,
	)
	print(f"Train samples: {len(train_df)}, Test samples: {len(eval_df)}\n")

	train_dataset = SurvivalTextDataset(
		train_df,
		tokenizer=tokenizer,
		max_length=cfg.max_length,
	)
	
	eval_dataset = SurvivalTextDataset(
		eval_df,
		tokenizer=tokenizer,
		max_length=cfg.max_length,
	)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
	train_loader_noshuffle = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
	eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
	base_config = AutoConfig.from_pretrained(cfg.model_name)
	text_encoder = BertForSurvival(base_config).to(device)

	print(f"Model loaded: {cfg.model_name}")
	print(f"Hidden size: {base_config.hidden_size}\n")
	
	encoder_param = list(text_encoder.encoder.parameters())
	head_param = list(text_encoder.survival_head.parameters())

	optimizer = torch.optim.AdamW([
		{'params': encoder_param, 'lr': 2e-5, 'weight_decay': 0.01},
		{'params': head_param, 'lr': 1e-3}
	])
	model = CoxPH(text_encoder, optimizer)

	total_steps = len(train_loader) * args.num_epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer, 
		num_warmup_steps = int(0.1 * total_steps),
		num_training_steps = total_steps
	)

	best_c_idx = -1
	
	history = []
	for epoch in range(args.num_epochs):

		model.net.train()
		train_loss = 0
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} / {args.num_epochs}")
		for x, y in pbar:
			x = {k: v.to(device) for k, v in x.items()}
			# check data type, if item can be deployed to gpu
			durations, events = [item.to(device) for item in y]

			optimizer.zero_grad() 
			output = model.net(**x)
			log_hazard = output["output"]

			loss = model.loss(log_hazard, durations, events)
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()
			train_loss += loss.item()
			pbar.set_postfix({'loss': f"{loss.item():.4f}"})
		avg_train_loss = train_loss / len(train_loader)
		print(f"avg train loss: {avg_train_loss:.4f}")


		# evaluate
		model.net.eval() 
		with torch.no_grad():

			all_preds = []
			all_durations = []
			all_events = []

			for x_val, y_val in eval_loader:
				x_val = {k: v.to(device) for k, v in x_val.items()}
				preds = model.net(**x_val)["output"]
				all_preds.append(preds.cpu())
				all_durations.append(y_val[0])
				all_events.append(y_val[1])

			preds = torch.cat(all_preds).numpy().flatten()
			durations = torch.cat(all_durations).numpy()
			events = torch.cat(all_events).numpy() 
			from lifelines.utils import concordance_index
			c_idx = concordance_index(durations, -preds, events)
			print(f"C_index: {c_idx:.4f}")

		# Self-computed integrated Brier score
		try:
			durations_train, events_train, g_train = _get_g_from_loader(train_loader_noshuffle, model.net, device)
			baseline_cumulative = _breslow_baseline_cumulative(durations_train, events_train, np.exp(g_train))
			durations_eval, events_eval, g_eval = _get_g_from_loader(eval_loader, model.net, device)
			time_grid = np.unique(durations_eval)
			time_grid = time_grid[time_grid >= 0]
			if len(time_grid) < 2:
				time_grid = np.linspace(float(durations_eval.min()), float(durations_eval.max()), max(20, len(time_grid)))
			surv_df = _coxph_surv_df(g_eval, baseline_cumulative, time_grid)
			ev_surv = EvalSurv(surv_df, durations_eval, events_eval, censor_surv="km")
			val_brier = _integrated_brier_manual(ev_surv, time_grid)
		except Exception:
			val_brier = float("nan")
		print(f"Brier: {val_brier:.4f}")

		epoch_metrics = {
			"epoch": epoch + 1,
			"train_loss": avg_train_loss,
			"val_c_idx": c_idx,
			"val_brier": val_brier,
		}
		print(epoch_metrics)
		history.append(epoch_metrics)
	history_df = pd.DataFrame(history)
	history_df.to_csv(f"runs/{args.run_name}/training_log.csv")


if __name__ == "__main__":
	main()