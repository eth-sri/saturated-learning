from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Sampler
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


@dataclass
class ScoreCollator(DataCollatorForLanguageModeling):
    def torch_call(self, features):
        scores = torch.tensor([f.pop("score") for f in features], dtype=torch.float32)
        group_ids = torch.tensor([f.pop("group_id") for f in features], dtype=torch.long)
        batch = super().torch_call(features)
        batch["score"] = scores
        batch["group_id"] = group_ids
        return batch


class GroupedBatchSampler(Sampler):
    def __init__(self, group_ids, batch_size):
        self.batch_size = batch_size
        groups = defaultdict(list)
        for i, g in enumerate(group_ids):
            groups[g].append(i)
        self.groups = list(groups.values())

    def __iter__(self):
        order = torch.randperm(len(self.groups)).tolist()
        batch = []
        for i in order:
            g = self.groups[i]
            if len(batch) + len(g) > self.batch_size and batch:
                yield batch
                batch = []
            batch.extend(g)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return sum(len(g) for g in self.groups) // self.batch_size


class RRHFTrainer(SFTTrainer):
    def __init__(self, *args, bt_reweight=False, rank_weight=1.0, **kw):
        super().__init__(*args, **kw)
        self.bt_reweight = bt_reweight
        self.rank_weight = rank_weight

    def get_train_dataloader(self):
        ds = self.train_dataset
        sampler = GroupedBatchSampler(ds["group_id"], self.args.per_device_train_batch_size)
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        scores = inputs.pop("score").float()
        group_ids = inputs.pop("group_id")
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        valid = (shift_labels != -100).float()
        log_p = -(nll * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)

        rank_losses, ft_losses = [], []
        for g in group_ids.unique():
            m = group_ids == g
            gs, gp, gnll, gv = scores[m], log_p[m], nll[m], valid[m]
            score_diff = gs.unsqueeze(1) - gs.unsqueeze(0)
            p_diff = gp.unsqueeze(1) - gp.unsqueeze(0)
            better = score_diff > 0
            if better.any():
                pairs = -p_diff[better]
                if self.bt_reweight:
                    pairs = pairs * torch.sigmoid(score_diff[better])
                rank_losses.append(pairs.sum())
            best = gs.argmax()
            ft_losses.append((gnll[best] * gv[best]).sum() / gv[best].sum().clamp(min=1.0))

        l_rank = torch.stack(rank_losses).mean() if rank_losses else log_p.new_zeros(())
        l_ft = torch.stack(ft_losses).mean()
        loss = self.rank_weight * l_rank + l_ft
        return (loss, outputs) if return_outputs else loss
