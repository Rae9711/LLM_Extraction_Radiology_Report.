import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
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
from pycox.models import PCHazard
from pycox.models.utils import pad_col, make_subgrid
from pycox.evaluation import EvalSurv
import torchtuples as tt
from tqdm import tqdm


class PCHazardTextDataset(Dataset):
	"""Dataset that returns (x, (idx_duration, event, interval_frac)) for PCHazard training."""

	def __init__(
		self,
		df: pd.DataFrame,
		tokenizer: AutoTokenizer,
		idx_durations: np.ndarray,
		events: np.ndarray,
		interval_frac: np.ndarray,
		max_length: int = 256,
	):
		self.df = df.reset_index(drop=True)
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.idx_durations = idx_durations  # (len(df),) int64
		self.events = events  # (len(df),) float32
		self.interval_frac = interval_frac  # (len(df),) float32
		self._input_list = []
		for idx in range(len(self.df)):
			row = self.df.iloc[idx]
			enc = self.tokenizer(
				str(row["text"]),
				padding="max_length",
				truncation=True,
				max_length=self.max_length,
				return_tensors="pt",
			)
			x = {
				"input_ids": enc["input_ids"].squeeze(0),
				"attention_mask": enc["attention_mask"].squeeze(0),
			}
			if "token_type_ids" in enc:
				x["token_type_ids"] = enc["token_type_ids"].squeeze(0)
			self._input_list.append(x)

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		x = self._input_list[idx]
		idx_dur = torch.tensor(self.idx_durations[idx], dtype=torch.long)
		ev = torch.tensor(self.events[idx], dtype=torch.float32)
		frac = torch.tensor(self.interval_frac[idx], dtype=torch.float32)
		return x, (idx_dur, ev, frac)


class EvalTextDataset(Dataset):
	"""Eval dataset: (x, (duration_orig, event_orig)) for EvalSurv."""

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

	def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
		row = self.df.iloc[idx]
		enc = self.tokenizer(
			str(row["text"]),
			padding="max_length",
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		x = {
			"input_ids": enc["input_ids"].squeeze(0),
			"attention_mask": enc["attention_mask"].squeeze(0),
		}
		if "token_type_ids" in enc:
			x["token_type_ids"] = enc["token_type_ids"].squeeze(0)
		duration = torch.tensor(row["duration"], dtype=torch.float32)
		event = torch.tensor(row["event"], dtype=torch.float32)
		return x, (duration, event)


def _collate_pchazard_batch(batch: List) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
	"""Collate train batch: (x_batch, (idx_durations, events, interval_frac))."""
	xs, ys = zip(*batch)
	x_batch = {k: torch.stack([d[k] for d in xs], dim=0) for k in xs[0].keys()}
	idx_durations = torch.stack([y[0] for y in ys])
	events = torch.stack([y[1] for y in ys])
	interval_frac = torch.stack([y[2] for y in ys])
	return x_batch, (idx_durations, events, interval_frac)


def _collate_eval_batch(batch: List) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
	"""Collate eval batch: (x_batch, (durations, events))."""
	xs, ys = zip(*batch)
	x_batch = {k: torch.stack([d[k] for d in xs], dim=0) for k in xs[0].keys()}
	durations = torch.stack([y[0] for y in ys])
	events = torch.stack([y[1] for y in ys])
	return x_batch, (durations, events)


class BertForPCHazard(PreTrainedModel):
	"""BERT encoder + MLP head outputting log-hazard per interval; forward returns (batch, out_features)."""

	config_class = AutoConfig

	def __init__(self, config, out_features: int, num_nodes=None, dropout=0.1, batch_norm=True):
		super().__init__(config)
		if num_nodes is None:
			num_nodes = [32, 32]
		self.encoder = AutoModel.from_config(config)
		hidden_size = config.hidden_size
		self.survival_head = tt.practical.MLPVanilla(
			hidden_size,
			num_nodes,
			out_features,
			batch_norm=batch_norm,
			dropout=dropout,
		)
		self.post_init()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		x=None,
		**kwargs,
	) -> Dict[str, torch.Tensor]:
		if x is not None:
			input_ids = x["input_ids"]
			attention_mask = x["attention_mask"]
			token_type_ids = x.get("token_type_ids")
		encoder_inputs = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
		}
		if token_type_ids is not None and hasattr(self.encoder, "embeddings") and hasattr(
			self.encoder.embeddings, "token_type_embeddings"
		):
			encoder_inputs["token_type_ids"] = token_type_ids
		outputs = self.encoder(**encoder_inputs)
		cls_emb = outputs.last_hidden_state[:, 0, :]
		phi = self.survival_head(cls_emb)
		return {"output": phi}


def predict_surv_df_from_loader(model: PCHazard, eval_loader: DataLoader, device: torch.device, sub: int = 1) -> pd.DataFrame:
	"""Build survival DataFrame by iterating eval_loader and applying PCHazard hazard/surv logic."""
	model.net.eval()
	all_phi = []
	with torch.no_grad():
		for x_batch, _ in eval_loader:
			x_batch = {k: v.to(device) for k, v in x_batch.items()}
			phi = model.net(**x_batch)["output"]
			all_phi.append(phi.cpu())
	phi_cat = torch.cat(all_phi, dim=0)
	n = phi_cat.shape[0]
	hazard = F.softplus(phi_cat).view(-1, 1).repeat(1, sub).view(n, -1).div(sub)
	hazard = pad_col(hazard, where="start")
	surv = hazard.cumsum(1).mul(-1).exp()
	surv_np = surv.numpy()
	index = None
	if model.duration_index is not None:
		index = make_subgrid(model.duration_index, sub)
	return pd.DataFrame(surv_np.transpose(), index=index)


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
	output_dir: str = "./outputs_pchazard"
	data_path: str = ""


def parse_args():
	parser = argparse.ArgumentParser(description="Train PCHazard survival model on text data")
	parser.add_argument(
		"--base_model",
		type=str,
		default="roberta-base",
		help="Pretrained model name or path (e.g., roberta-base, bert-base-uncased)",
	)
	parser.add_argument(
		"--data_path",
		type=str,
		required=True,
		help="Path to the CSV file containing the dataset",
	)
	parser.add_argument(
		"--run_name",
		type=str,
		required=True,
		help="Name for this training run",
	)
	parser.add_argument("--batch_size", type=int, default=32, help="Training and evaluation batch size")
	parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
	parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
	parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
	parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
	parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset to use for testing")
	parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("--logging_steps", type=int, default=50, help="Log metrics every N steps")
	parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate model every N steps")
	parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
	parser.add_argument(
		"--num_durations",
		type=int,
		default=30,
		help="Number of duration intervals for PCHazard label transform",
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

	print("=" * 60)
	print(f"PCHazard Training Configuration for run: {args.run_name}")
	print("=" * 60)
	print(f"Model: {cfg.model_name}")
	print(f"Data: {cfg.data_path}")
	print(f"Output Directory: {cfg.output_dir}")
	print(f"Batch Size: {cfg.train_batch_size}")
	print(f"Learning Rate: {cfg.learning_rate}")
	print(f"Epochs: {cfg.num_train_epochs}")
	print(f"Max Length: {cfg.max_length}")
	print(f"Num durations: {args.num_durations}")
	print("=" * 60 + "\n")

	tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

	print(f"Loading data from {cfg.data_path}...")
	df = pd.read_csv(cfg.data_path)
	print(f"Total samples: {len(df)}")

	train_df, eval_df = train_test_split(
		df,
		test_size=cfg.test_size,
		random_state=cfg.random_state,
		shuffle=True,
	)
	train_df = train_df.reset_index(drop=True)
	eval_df = eval_df.reset_index(drop=True)
	print(f"Train samples: {len(train_df)}, Test samples: {len(eval_df)}\n")

	# PCHazard label transform
	labtrans = PCHazard.label_transform(args.num_durations)
	durations_train = train_df["duration"].values
	events_train = train_df["event"].values
	idx_durations_train, events_train, interval_frac_train = labtrans.fit_transform(durations_train, events_train)
	durations_eval = eval_df["duration"].values
	events_eval = eval_df["event"].values
	idx_durations_eval, events_eval_t, interval_frac_eval = labtrans.transform(durations_eval, events_eval)
	# Keep original durations_eval, events_eval for EvalSurv (unchanged)
	del events_eval_t, interval_frac_eval

	train_dataset = PCHazardTextDataset(
		train_df,
		tokenizer=tokenizer,
		idx_durations=idx_durations_train,
		events=events_train,
		interval_frac=interval_frac_train,
		max_length=cfg.max_length,
	)
	eval_dataset = EvalTextDataset(
		eval_df,
		tokenizer=tokenizer,
		max_length=cfg.max_length,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=0,
		collate_fn=_collate_pchazard_batch,
	)
	eval_loader = DataLoader(
		eval_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=_collate_eval_batch,
	)

	base_config = AutoConfig.from_pretrained(cfg.model_name)
	out_features = labtrans.out_features
	net = BertForPCHazard(base_config, out_features=out_features).to(device)

	print(f"Model loaded: {cfg.model_name}")
	print(f"Hidden size: {base_config.hidden_size}, out_features: {out_features}\n")

	encoder_param = list(net.encoder.parameters())
	head_param = list(net.survival_head.parameters())
	optimizer = torch.optim.AdamW([
		{"params": encoder_param, "lr": 2e-5, "weight_decay": 0.01},
		{"params": head_param, "lr": 1e-3},
	])
	model = PCHazard(net, optimizer, duration_index=labtrans.cuts)

	total_steps = len(train_loader) * args.num_epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=int(0.1 * total_steps),
		num_training_steps=total_steps,
	)

	history = []
	for epoch in range(args.num_epochs):
		model.net.train()
		train_loss = 0
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} / {args.num_epochs}")
		for x_batch, (idx_durations, events, interval_frac) in pbar:
			x_batch = {k: v.to(device) for k, v in x_batch.items()}
			idx_durations = idx_durations.to(device)
			events = events.to(device)
			interval_frac = interval_frac.to(device)

			optimizer.zero_grad()
			phi = model.net(**x_batch)["output"]
			loss = model.loss(phi, idx_durations, events, interval_frac)
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()
			train_loss += loss.item()
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})
		avg_train_loss = train_loss / len(train_loader)
		print(f"avg train loss: {avg_train_loss:.4f}")

		# Evaluate: predict surv DataFrame then EvalSurv concordance
		surv_df = predict_surv_df_from_loader(model, eval_loader, device, sub=model.sub)
		ev = EvalSurv(surv_df, durations_eval, events_eval, censor_surv="km")
		c_idx = ev.concordance_td("antolini")
		print(f"C_index: {c_idx:.4f}")
		time_grid = np.linspace(durations_eval.min(), durations_eval.max(), 100)
		scores = ev.brier_score(time_grid)
		scores = scores.values
		scores = np.asarray(scores, dtype=np.float64)
		val_brier = float(np.trapz(scores, time_grid) / (time_grid[-1] - time_grid[0]))
		epoch_metrics = {
			"epoch": epoch + 1,
			"train_loss": avg_train_loss,
			"val_c_idx": c_idx,
			"brier_score": val_brier,
		}
		print(epoch_metrics)
		history.append(epoch_metrics)

	history_df = pd.DataFrame(history)
	history_df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)


if __name__ == "__main__":
	main()
