# this files helps to convert the model from causal to sequence modelling part 
import torch 
import torch.nn as nn

import os
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, PreTrainedModel

# take the base class as pretrained model 
class ConvertedModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        num_labels = kwargs.get('num_labels', 1)
        model = kwargs.get('model', "google/gemma-3-270m-it")
        self.model_name = model
        self.num_labels = num_labels

        self.FILE_PATH = os.path.dirname(os.path.dirname(__file__))

        # Device selection (MPS > CUDA > CPU)
        if torch.backends.mps.is_available():
            self.device_str = 'mps'
        elif torch.cuda.is_available():
            self.device_str = 'cuda'
        else:
            self.device_str = 'cpu'

        # Load the base CausalLM model
        self.lm_model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.FILE_PATH)
        for params in self.lm_model.parameters():
            params.requires_grad = False

        # Replace the language modeling head with a classification head
        logits_count = self.lm_model.lm_head.in_features
        self.lm_model.lm_head = nn.Linear(
            in_features=logits_count,
            out_features=self.num_labels,
            bias=True,
            device=self.device_str
        )
        for params in self.lm_model.lm_head.parameters():
            params.requires_grad = True



    # def forward(self, input_ids, attention_mask=None):
    #     # Get hidden states from the base model
    #     outputs = self.lm_model.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         output_hidden_states=False
    #     )

    #     # Take last token representation for classification
    #     last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
    #     pooled = last_hidden[:, -1, :]  # last token

    #     # Classify using the replaced head
    #     logits = self.lm_model.lm_head(pooled)
    #     return {"logits": logits}



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs
    ):
        # Hidden states
        outputs = self.lm_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        # Last hidden state
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        pooled = last_hidden[:, -1, :]  # last token

        # Classification head
        logits = self.lm_model.lm_head(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            return (logits,) if loss is None else (loss, logits)

        return {"loss": loss, "logits": logits}


    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.lm_model.lm_head.state_dict(), os.path.join(save_directory, "lm_head.pt"))
