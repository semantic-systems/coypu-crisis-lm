from typing import Optional, Any, Tuple
from dataclasses import dataclass

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """This custom data collator overwrites the mlm functionality to support uniform masking
    instead of the 80/10/10 method used in the original BERT implementation. Motivated by this:
    https://twitter.com/gaotianyu1350/status/1493919318668713986?s=20&t=E9XItu9_8MAs2TSKHmFByA."""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    uniform_masking: bool = False
    return_tensors: str = "pt"

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[
        Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val
                in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        if self.uniform_masking:
            # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.mask_token)
        else:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(
                torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels