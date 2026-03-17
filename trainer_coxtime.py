import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

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
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.models.data import make_at_risk_dict, sample_alive_from_dates
from pycox.evaluation import EvalSurv
from tqdm import tqdm


class SurvivalTextDatasetWithTime(Dataset):
	"""Eval dataset that returns (x, (duration_orig, event, duration_scaled)) for CoxTime eval."""

	def __init__(
		self,
		df: pd.DataFrame,
		tokenizer: AutoTokenizer,
		max_length: int = 256,
		durations_scaled: np.ndarray = None,
	):
		self.df = df.reset_index(drop=True)
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.durations_scaled = durations_scaled  # (len(df),) aligned with df

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		row = self.df.iloc[idx]
		text = str(row["text"])
		enc = self.tokenizer(
			text,
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
		if self.durations_scaled is not None:
			time_scaled = torch.tensor(
				np.float32(self.durations_scaled[idx]).reshape(1, 1),
				dtype=torch.float32,
			)
		else:
			time_scaled = torch.tensor(
				np.float32(row["duration"]).reshape(1, 1),
				dtype=torch.float32,
			)
		return x, (duration, event, time_scaled)


def _collate_eval_with_time(batch: List) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
	"""Collate eval batch: (x_batch with 'time' key, (durations, events))."""
	xs, ys = zip(*batch)
	x_batch = {k: torch.stack([d[k] for d in xs], dim=0) for k in xs[0].keys()}
	durations = torch.stack([y[0] for y in ys])
	events = torch.stack([y[1] for y in ys])
	time_scaled = torch.cat([y[2] for y in ys], dim=0)  # (batch, 1)
	x_batch["time"] = time_scaled
	return x_batch, (durations, events)


class CoxTimeFlatDataset(Dataset):
	"""Per-sample (x with time, event) for Breslow baseline; no case-control."""

	def __init__(
		self,
		df: pd.DataFrame,
		tokenizer: AutoTokenizer,
		durations_scaled: np.ndarray,
		events: np.ndarray,
		max_length: int = 256,
	):
		self.df = df.reset_index(drop=True)
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.durations_scaled = durations_scaled
		self.events = events
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

	def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
		x = {k: v.clone() for k, v in self._input_list[idx].items()}
		t = torch.tensor(np.float32(self.durations_scaled[idx]).reshape(1, 1), dtype=torch.float32)
		x["time"] = t
		ev = torch.tensor(self.events[idx], dtype=torch.float32)
		return x, ev


def _collate_flat_batch(batch: List) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
	xs, events = zip(*batch)
	x_batch = {k: torch.stack([d[k] for d in xs], dim=0) for k in xs[0].keys()}
	x_batch["time"] = torch.cat([d["time"] for d in xs], dim=0)
	events_batch = torch.stack(events)
	return x_batch, events_batch


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


def _get_g_from_loader(
	loader: DataLoader,
	model: nn.Module,
	device: torch.device,
	flat_time_in_batch: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return (durations, events, g). If flat_time_in_batch, durations from x_batch['time']."""
	model.eval()
	durations_list, events_list, g_list = [], [], []
	with torch.no_grad():
		for x_batch, second in loader:
			x_batch = {k: v.to(device) for k, v in x_batch.items()}
			if flat_time_in_batch:
				durations_list.append(x_batch["time"].squeeze(-1).cpu().numpy())
				events_list.append(second.cpu().numpy())
			else:
				durations, events = second
				durations_list.append(durations.numpy())
				events_list.append(events.numpy())
			g = model(**x_batch)["output"].squeeze(-1).cpu().numpy()
			g_list.append(g)
	model.train()
	return np.concatenate(durations_list), np.concatenate(events_list), np.concatenate(g_list)


def _coxtime_surv_df(
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


def _collate_coxtime_batch(batch):
	"""Collate list of ((x_case, time_case), (x_control, time_case)) into batched tensors for CoxTime."""
	x_cases, time_cases = zip(*[b[0] for b in batch])
	# x_cases[i] is list of 1 dict; time_cases[i] is (1, 1)
	flat_cases = [d for lst in x_cases for d in lst]
	case_batch = {
		k: torch.stack([d[k] for d in flat_cases], dim=0)
		for k in flat_cases[0].keys()
	}
	time_case_batch = torch.cat(time_cases, dim=0)  # (batch_size, 1)
	case_batch["time"] = time_case_batch

	x_controls = [b[1][0] for b in batch]  # list of n_control lists of dicts
	time_controls = [b[1][1] for b in batch]  # same time as case per sample
	n_control = len(x_controls[0])
	control_batches = []
	for c in range(n_control):
		flat_c = [
			x_controls[i][c][j]
			for i in range(len(x_controls))
			for j in range(len(x_controls[i][c]))
		]
		ctrl_batch = {
			k: torch.stack([d[k] for d in flat_c], dim=0)
			for k in flat_c[0].keys()
		}
		# same time as case for this batch (each control paired with case)
		ctrl_batch["time"] = time_case_batch
		control_batches.append(ctrl_batch)
	return case_batch, tuple(control_batches)


class CoxTimeTextDataset(Dataset):
	"""Case-control dataset for CoxTime: sorted by scaled duration, cases (event=1) with n_control at-risk controls; each sample carries time (scaled duration)."""

	def __init__(
		self,
		df: pd.DataFrame,
		tokenizer: AutoTokenizer,
		max_length: int = 256,
		n_control: int = 1,
		durations_scaled: np.ndarray = None,
	):
		self.n_control = n_control
		if durations_scaled is None:
			durations_scaled = df["duration"].values.astype(np.float32)
		# Sort by scaled duration and keep order
		df = df.copy()
		df["_duration_scaled"] = durations_scaled
		df = df.sort_values("_duration_scaled").reset_index(drop=True)
		durations_scaled = df["_duration_scaled"].values
		self.df = df.drop(columns=["_duration_scaled"])
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.durations_scaled = durations_scaled
		events = self.df["event"].values.astype(np.float32)
		self.at_risk_dict = make_at_risk_dict(durations_scaled)
		self.case_indices = np.where(events == 1)[0]
		self.durations_cases = durations_scaled[self.case_indices]  # (n_cases,)
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
		return len(self.case_indices)

	def __getitem__(self, idx):
		if isinstance(idx, (int, np.integer)):
			idx = [idx]
		pos = self.case_indices[idx]
		durations_at = self.durations_cases[idx]
		x_case = [self._input_list[i] for i in pos]
		# time for case: (len(idx), 1)
		time_case = torch.tensor(
			durations_at.reshape(-1, 1).astype(np.float32),
			dtype=torch.float32,
		)
		control_idx = sample_alive_from_dates(durations_at, self.at_risk_dict, self.n_control)
		x_control = [
			[self._input_list[control_idx[j, c]] for j in range(control_idx.shape[0])]
			for c in range(self.n_control)
		]
		return (x_case, time_case), (x_control, time_case)


class BertForCoxTime(PreTrainedModel):
	"""BERT encoder + MLPVanillaCoxTime head; forward(input_ids, attention_mask, time) -> {output}."""

	config_class = AutoConfig

	def __init__(self, config, num_nodes=None, dropout=0.1, batch_norm=True):
		super().__init__(config)
		if num_nodes is None:
			num_nodes = [32, 32]
		self.encoder = AutoModel.from_config(config)
		hidden_size = config.hidden_size
		self.survival_head = MLPVanillaCoxTime(
			hidden_size,
			num_nodes,
			batch_norm=batch_norm,
			dropout=dropout,
		)
		self.post_init()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		time=None,
		**kwargs,
	) -> Dict[str, torch.Tensor]:
		if time is None:
			raise ValueError("CoxTime requires 'time' input (scaled duration) of shape (batch_size, 1).")
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
		output = self.survival_head(cls_emb, time)
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
	output_dir: str = "./outputs_coxtime"
	data_path: str = ""


def parse_args():
	parser = argparse.ArgumentParser(description="Train CoxTime survival model on text data")
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
		"--n_control",
		type=int,
		default=1,
		help="Number of control samples per case for CoxTime",
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
	print(f"CoxTime Training Configuration for run: {args.run_name}")
	print("=" * 60)
	print(f"Model: {cfg.model_name}")
	print(f"Data: {cfg.data_path}")
	print(f"Output Directory: {cfg.output_dir}")
	print(f"Batch Size: {cfg.train_batch_size}")
	print(f"Learning Rate: {cfg.learning_rate}")
	print(f"Epochs: {cfg.num_train_epochs}")
	print(f"Max Length: {cfg.max_length}")
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
	print(f"Train samples: {len(train_df)}, Test samples: {len(eval_df)}\n")

	# Label transform for CoxTime (scale durations)
	labtrans = CoxTime.label_transform()
	durations_train = train_df["duration"].values
	events_train = train_df["event"].values
	durations_train_scaled, events_train = labtrans.fit_transform(durations_train, events_train)
	durations_eval = eval_df["duration"].values
	events_eval = eval_df["event"].values
	durations_eval_scaled, events_eval = labtrans.transform(durations_eval, events_eval)

	# Align scaled durations with dataframe index (train_df/eval_df are already reset by train_test_split in order)
	train_df = train_df.reset_index(drop=True)
	eval_df = eval_df.reset_index(drop=True)

	train_dataset = CoxTimeTextDataset(
		train_df,
		tokenizer=tokenizer,
		max_length=cfg.max_length,
		n_control=args.n_control,
		durations_scaled=durations_train_scaled,
	)

	eval_dataset = SurvivalTextDatasetWithTime(
		eval_df,
		tokenizer=tokenizer,
		max_length=cfg.max_length,
		durations_scaled=durations_eval_scaled,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=0,
		collate_fn=_collate_coxtime_batch,
	)
	eval_loader = DataLoader(
		eval_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=_collate_eval_with_time,
	)
	flat_train_dataset = CoxTimeFlatDataset(
		train_df, tokenizer, durations_train_scaled, events_train, max_length=cfg.max_length,
	)
	flat_train_loader = DataLoader(
		flat_train_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=_collate_flat_batch,
	)

	base_config = AutoConfig.from_pretrained(cfg.model_name)
	text_encoder = BertForCoxTime(base_config).to(device)

	print(f"Model loaded: {cfg.model_name}")
	print(f"Hidden size: {base_config.hidden_size}\n")

	encoder_param = list(text_encoder.encoder.parameters())
	head_param = list(text_encoder.survival_head.parameters())
	optimizer = torch.optim.AdamW([
		{"params": encoder_param, "lr": 2e-5, "weight_decay": 0.01},
		{"params": head_param, "lr": 1e-3},
	])
	model = CoxTime(text_encoder, optimizer, labtrans=labtrans)

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
		for x_case, x_control in pbar:
			x_case = {k: v.to(device) for k, v in x_case.items()}
			x_control = tuple({k: v.to(device) for k, v in ctrl.items()} for ctrl in x_control)

			optimizer.zero_grad()
			g_case = model.net(**x_case)["output"]
			g_control = tuple(model.net(**ctrl)["output"] for ctrl in x_control)
			loss = model.loss(g_case, g_control)
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()
			train_loss += loss.item()
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})
		avg_train_loss = train_loss / len(train_loader)
		print(f"avg train loss: {avg_train_loss:.4f}")

		# Evaluate with scaled duration as time
		model.net.eval()
		all_preds = []
		all_durations = []
		all_events = []
		with torch.no_grad():
			for x_val, (durations_orig, events_orig) in eval_loader:
				x_val = {k: v.to(device) for k, v in x_val.items()}
				preds = model.net(**x_val)["output"]
				all_preds.append(preds.cpu())
				all_durations.append(durations_orig)
				all_events.append(events_orig)

		preds = torch.cat(all_preds).numpy().flatten()
		durations = torch.cat(all_durations).numpy()
		events = torch.cat(all_events).numpy()
		from lifelines.utils import concordance_index

		c_idx = concordance_index(durations, -preds, events)
		print(f"C_index: {c_idx:.4f}")

		# Self-computed integrated Brier score (no pycox simps)
		try:
			_, _, g_train = _get_g_from_loader(flat_train_loader, model.net, device, flat_time_in_batch=True)
			baseline_cumulative = _breslow_baseline_cumulative(
				durations_train_scaled, events_train, np.exp(g_train),
			)
			_, _, g_eval = _get_g_from_loader(eval_loader, model.net, device, flat_time_in_batch=False)
			time_grid = np.unique(durations_eval_scaled)
			time_grid = time_grid[time_grid >= 0]
			if len(time_grid) < 2:
				time_grid = np.linspace(
					float(durations_eval_scaled.min()), float(durations_eval_scaled.max()),
					max(20, len(time_grid)),
				)
			surv_df = _coxtime_surv_df(g_eval, baseline_cumulative, time_grid)
			ev_surv = EvalSurv(surv_df, durations_eval_scaled, events_eval, censor_surv="km")
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
	history_df.to_csv(f"runs/{args.run_name}/training_log.csv", index=False)


if __name__ == "__main__":
	main()
