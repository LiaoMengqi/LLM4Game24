import copy
from dataclasses import dataclass
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .reward_model import RewardModel4Game24


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Samples:
    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    case: Optional[list[int]]


@dataclass
class Experience:
    sequences: torch.Tensor
    old_log_probs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    ref_log_probs: Optional[torch.Tensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.old_log_probs = to(self.old_log_probs, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.ref_log_probs = to(self.ref_log_probs, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self


class GRPOActor(nn.Module):
    def __init__(self, model):
        super(GRPOActor, self).__init__()
        self.model = model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs):
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_return_sequences": kwargs.get("num_return_sequences", 1),
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
            "max_new_tokens": kwargs.get("max_new_tokens", 2048),
        }
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1: -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(self, sequences,
                attention_mask: torch.Tensor,
                num_actions: Optional[Union[int, list[int]]] = None):
        def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
            return log_probs_labels.squeeze(-1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        if num_actions is None:
            return output

        output["logits"] = output["logits"].to(torch.float32)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])
        action_log_probs = log_probs[:, -num_actions:]
        return action_log_probs


class GRPOTrainer4Game24:
    """
    Trainer for Group Relative Policy Optimization (GRPO).
    """

    def __init__(self,
                 policy_model: nn.Module,
                 reward_model: RewardModel4Game24,
                 tokenizer,
                 args):
        """
        Initialize the GRPO Trainer.

        Args:
            policy_model: The policy model (π_θ).
            reward_model: The reward model (r_φ).
            optimizer: Optimizer for the policy model.
            tokenizer: Tokenizer for input prompts.
            args: Training arguments (including hyperparameters ε, β, μ, group_size, etc.).
        """
        self.actor = GRPOActor(policy_model.cuda())
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.args = args
        self.act_device = f"cuda:{args.act_device}"
        self.ref_device = f"cuda:{args.ref_device}"
        self.reference_model = None  # π_ref
        self.optimizer = AdamW(self.actor.parameters(), lr=self.args.lr)
        self.momentum = 0
        self.mean = None
        self.std = None

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """

        self.actor.eval()
        self.reference_model.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        if self.args.grpo_iterations > 1:
            old_log_probs = self.actor(sequences, attention_mask=attention_mask, num_actions=num_actions).cpu()
        else:
            old_log_probs = None

        # ref log probs
        ref_log_probs = self.reference_model(sequences.to(self.ref_device), attention_mask.to(self.ref_device),
                                             num_actions)

        response_sentences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        cases = [samples.case] * len(response_sentences)
        r, c = self.reward_model(response_sentences, cases)

        info = {
            # "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "correct": c.sum(),
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()

        return Experience(
            sequences=sequences.cpu(),
            old_log_probs=old_log_probs,
            attention_mask=attention_mask.cpu(),
            action_mask=action_mask.cpu(),
            info=info,
            ref_log_probs=ref_log_probs.cpu()
            # kl,
        )

    def train(self, dataloader: DataLoader):
        """
        Train the policy model using GRPO.

        Args:
            dataloader: DataLoader for task prompts.
        """
        self.reference_model = self._clone_model(self.actor, self.ref_device)  # π_ref ← π_θ

        loss_list, acc_list = [], []
        tqdm_bar = tqdm(total=self.args.max_step)
        step = 1
        for epoch in range(self.args.max_epoch):
            for data_index, batch in enumerate(dataloader):
                correct: Union[torch.Tensor, int] = 0
                if step > self.args.max_step:
                    break
                # print(f"  Step {step + 1}/{len(dataloader)}")
                prompts = batch['prompts']
                samples_list = self.generate_samples(prompts, self.actor)
                experiences = []
                for i, samples in enumerate(samples_list):
                    samples.case = batch['cases'][i]
                    experiences.append(self.make_experience(samples).to_device("cpu"))
                    correct += experiences[-1].info["correct"]

                # Perform GRPO updates
                for grpo_iteration in range(self.args.grpo_iterations):
                    loss = self.update_policy(experiences)
                    loss_list.append(loss)

                tqdm_bar.update(1)
                acc = correct.item() / (len(batch['prompts']) * self.args.group_size)
                acc_list.append(acc)
                tqdm_bar.set_postfix(loss=loss_list[-1], acc=acc)
                print()
                if (step) % self.args.save_step == 0:
                    self.actor.model.save_pretrained(
                        f"./checkpoint/{self.args.save_dir}_{(step) // self.args.save_step}/")
                step += 1
            if step > self.args.max_step:
                break

        return loss_list, acc_list

    def _clone_model(self, model, device="cpu"):
        """
        Clone a model to create a reference model (π_ref or π_old).
        """
        cloned_model = copy.deepcopy(model)
        cloned_model = cloned_model.to(device)
        return cloned_model

    # def meke_experience(self, prompts, outputs, log_probs_old, advantages):
    #     pass

    @torch.no_grad()
    def generate_samples(self, prompts, policy_model):
        """
        Sample G outputs for each prompt using the old policy model.
        """
        samples_list = []
        for prompt in prompts:
            # Tokenize prompt
            model_input = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.act_device)

            # Generate G outputs
            sequences, attention_mask, action_mask = policy_model.generate(model_input.input_ids,
                                                                           max_new_tokens=self.args.max_len,
                                                                           num_return_sequences=self.args.group_size,
                                                                           do_sample=True,
                                                                           top_p=self.args.top_p,
                                                                           temperature=self.args.temperature,
                                                                           attention_mask=model_input.attention_mask,
                                                                           pad_token_id=self.tokenizer.pad_token_id,
                                                                           eos_token_id=self.tokenizer.eos_token_id
                                                                           )
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                case=None
            )
            samples_list.append(samples)

        return samples_list

    def update_policy(self, experiences: List[Experience]):
        """
        Compute the GRPO loss for a batch.
        """

        def compute_approx_kl(
                log_probs: torch.Tensor,
                log_probs_base: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                use_kl_estimator_k3: bool = False,
        ) -> torch.Tensor:

            log_ratio = log_probs.float() - log_probs_base.float()
            if action_mask is not None:
                log_ratio = log_ratio * action_mask

            if use_kl_estimator_k3:
                log_ratio = -log_ratio
                log_ratio = log_ratio.exp() - 1 - log_ratio

            return log_ratio

        @torch.no_grad()
        def compute_advantages(reward):
            mean = reward.mean()
            std = reward.std()
            if self.mean is None:
                self.mean = mean
            else:
                self.mean = self.mean * self.momentum + mean * (1 - self.momentum)
            if self.std is None:
                self.std = std
            else:
                self.std = self.std * self.momentum + std * (1 - self.momentum)
            self.momentum = min(self.args.max_momentum, self.momentum + 0.1)
            return (reward - self.mean) / (self.std + 1e-4)

        class packed_experience:
            def __init__(self, experiences):
                self.sequences = experiences.sequences
                self.attention_mask = experiences.attention_mask
                self.old_log_probs = experiences.old_log_probs
                self.action_mask = experiences.action_mask
                self.advantages = compute_advantages(experience.info['reward']).unsqueeze(1)
                self.ref_log_probs = experiences.ref_log_probs

            def get_experiences(self, start, end, device):
                if self.old_log_probs is not None:
                    old_log_probs = self.old_log_probs[start:end].to(device)
                else:
                    old_log_probs = None
                return (self.sequences[start:end].to(device), self.attention_mask[start:end].to(device),
                        old_log_probs, self.action_mask[start:end].to(device),
                        self.advantages[start:end].to(device), self.ref_log_probs[start:end].to(device))

        self.optimizer.zero_grad()
        total_loss = 0
        for experience in experiences:
            # Compute loss for a group
            num_actions = experience.info['num_actions']

            group_experience = packed_experience(experience)

            mini_batch_num = self.args.group_size // self.args.mini_batch_size + (
                0 if self.args.group_size % self.args.mini_batch_size == 0 else 1)

            for i in range(mini_batch_num):
                sequences, attention_mask, old_log_probs, action_mask, advantages, ref_log_probs = group_experience.get_experiences(
                    i * self.args.mini_batch_size, min((i + 1) * self.args.mini_batch_size, self.args.group_size),
                    device=self.act_device)

                new_log_probs = self.actor(sequences, attention_mask=attention_mask, num_actions=num_actions)

                if old_log_probs is None:
                    old_log_probs = new_log_probs.clone().detach()

                ratios = torch.exp(new_log_probs - old_log_probs)
                clipped_ratios = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon)

                surrogate_loss = torch.min(ratios * advantages, clipped_ratios * advantages)

                surrogate_loss = (surrogate_loss * action_mask).sum(dim=-1, keepdim=True)
                kl_loss = compute_approx_kl(new_log_probs, ref_log_probs, action_mask=action_mask).sum(dim=-1,
                                                                                                       keepdim=True)

                # Combine surrogate loss and KL divergence
                group_loss = (-surrogate_loss + self.args.beta * kl_loss) / action_mask.sum(dim=-1, keepdim=True)
                group_loss = group_loss.sum() / (self.args.group_size * len(experiences))
                group_loss.backward()
                total_loss += group_loss.item()
        self.optimizer.step()
        return total_loss
