import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaConfig, RobertaModel, XLMRobertaConfig


def z_norm(inputs):  # Batch Normalization
    mean = inputs.mean(0, keepdim=True)
    var = inputs.var(0, unbiased=False, keepdim=True)
    return (inputs - mean) / torch.sqrt(var + 1e-9)

class bertCNN(nn.Module):
    def __init__(self, embed_model, dropout=0.2, kernel_num=4, kernel_sizes=[4, 5, 6], num_labels=4):
        super().__init__()
        self.num_labels = num_labels
        self.embed = embed_model
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (k, self.embed.config.hidden_size)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(kernel_sizes) * kernel_num, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, freeze=True, is_norm=None):
        if freeze:
            with torch.no_grad():
                output = self.embed(input_ids, attention_mask, token_type_ids)[0]
        else:
            # freeze_layers = 6
            # for i, child in enumerate(self.embed.children()):
            #     if i <= freeze_layers:
            #         for param in child.parameters():
            #             param.requires_grad = False
            output = self.embed(input_ids, attention_mask, token_type_ids)[0]
        if is_norm:
            output = z_norm(output)
        output = output.unsqueeze(1)
        output = [nn.functional.relu(conv(output)).squeeze(3) for conv in self.convs]
        output = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        logits = self.classifier(output)
        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits



class bertDPCNN(nn.Module):
    def __init__(self,embed_model,num_filters=100, num_labels=4):
        super().__init__()
        self.num_labels=num_labels
        self.embed = embed_model
        self.conv_region = nn.Conv2d(1, num_filters, (3, self.embed.config.hidden_size))
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.classifer = nn.Linear(num_filters, self.num_labels)

    def forward(self,  input_ids, attention_mask, token_type_ids=None, labels=None, freeze=True, is_norm=None):
        if freeze:
            with torch.no_grad():
                output = self.embed(input_ids, attention_mask, token_type_ids)[0]
        else:
            output = self.embed(input_ids, attention_mask, token_type_ids)[0]
        if is_norm:
            output = z_norm(output)
        output = output.unsqueeze(1)
        output = self.conv_region(output)
        output = self.padding1(output)
        output = self.relu(output)
        output = self.conv(output)
        output = self.padding1(output)
        output = self.relu(output)
        output = self.conv(output)
        while output.size()[2] >= 2:
            output = self._block(output)
        output = output.squeeze()
        logits = self.classifer(output)
        if len(logits.shape)==1:
            logits=logits.reshape(1, self.num_labels)

        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = nn.functional.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = nn.functional.relu(x)
        x = self.conv(x)

        x = x + px
        return x


class basicBert(nn.Module):
    def __init__(self,embed_model, dropout=0.1, num_labels=3, freeze = True):
        super().__init__()
        self.embed = embed_model
        if freeze:
            for p in self.embed.parameters():
                p.requires_grad = False
        self.dropout=nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, freeze=True):
        output = self.embed(input_ids, attention_mask, token_type_ids)
        output=self.dropout(output[1])
        output=self.classifier(output)
        return output


"""
The following code is adapted from huggingface transformers:
https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
https://huggingface.co/transformers/_modules/transformers/modeling_xlm_roberta.html
"""


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, is_norm):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if is_norm:
            first_token_tensor = z_norm(first_token_tensor)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, freeze = False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            is_norm=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        pooled_output = self.pooler(sequence_output, is_norm)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, is_norm, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if is_norm:
            x = z_norm(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, freeze = False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        
        if freeze:
            for p in self.roberta.parameters():
                p.requires_grad = False
            
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_norm=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, is_norm=is_norm)
        outputs = (logits,) + outputs[2:]
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return outputs


class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    