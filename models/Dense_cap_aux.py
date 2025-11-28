# import random
import logging
# from abc import ABC

import torch
import math
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig, GPT2Config, GPT2LMHeadModel
# from models.position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model

import contextlib
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


logger = logging.getLogger(__name__)


def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))

def position_embedding(max_len: int, d_model: int):
    position_embedding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    position_embedding[:, 0::2] = torch.sin(position * div_term)
    position_embedding[:, 1::2] = torch.cos(position * div_term)
    return position_embedding


class DCA(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.end_sym = config.model.end_sym
        self.system_path = config.model.system_path
        self.instruction_path = config.model.instruction_path
        self.role = config.model.role
        self.add_scene_token = config.model.add_scene_token
        self.train_emb = config.model.train_emb
        self.input_mask3d_dim = config.model.input_mask3d_dim
        self.input_dim = config.model.input_dim
        self.img_input_dim = config.model.img_input_dim
        self.scene_dim = config.model.scene_dim
        self.pos_dim = config.model.pos_dim
        self.max_obj_num = config.model.max_obj_num
        self.add_pos_emb = config.model.add_pos_emb
        self.add_box_emb = config.model.add_box_emb
        self.add_sid_eid = config.model.add_sid_eid
        self.add_objlabel = config.model.add_objlabel
        self.add_layer_norm = config.model.add_layer_norm
        self.add_mask_token = config.model.add_mask_token
        self.add_mask3d_token = config.model.add_mask3d_token

        self.debug = config.debug
        
        if self.config.only_train_dca:
            self.llama_model = None
            self.llama_dim = 4096
        else:
            logger.info('Loading LLaMA')
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))
            
                lora_target_modules = find_linear_layers(self.llama_model, config.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                # we use additional head to handle newly added tokens.
                # search OBJ_lm_head in .modeling_llama.py for details.
                self.llama_model.model.lm_head.weight.requires_grad = False # True
                self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.model.embed_tokens.weight.data = self.llama_model.model.model.embed_tokens.weight.data.float()
                
                for name, param in self.llama_model.named_parameters():
                    if 'extra_adapter' in name:
                        param.requires_grad = True
                
                # self.llama_model.print_trainable_parameters()
            else:
                self.llama_model.lm_head.weight.requires_grad = False # True
                # self.llama_model.lm_head.weight.data = self.llama_model.lm_head.weight.data.float()
                self.llama_model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.embed_tokens.weight.data = self.llama_model.model.embed_tokens.weight.data.float()
            
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

            objid_tokens = []
            for i in range(150):
                objid_tokens.append(f"<OBJ{i:03}>")
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
            self.llama_tokenizer.add_tokens(['<click>'], special_tokens=True)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

            self.llama_dim = self.llama_model.config.hidden_size
            logger.info('Loading LLAMA Done')
        
        gpt2_config = GPT2Config(
                n_positions=64,
                n_embd=self.llama_dim,
                n_layer=2,
                n_head=4,
                bos_token_id=50256,
                eos_token_id=50256,
                add_cross_attention=False,
            )
        self.transformer = GPT2LMHeadModel(config=gpt2_config)
        self.transformer.transformer.wpe = nn.Embedding.from_pretrained(
            position_embedding(64, self.llama_dim)
        )
        self.loss_fct = CrossEntropyLoss()
        self.object_pos_proj = nn.Sequential(
            nn.Linear(self.input_mask3d_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim + self.input_mask3d_dim, self.llama_dim),
            # nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )

    def get_objid_embeds(self):
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_start_idx+150]
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_start_idx+150]
        return objid_embeds
    
    def llama_embed_tokens(self, token_ids):
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def get_text_emb(self, token_id):
        embeds = self.llama_embed_tokens(token_id)
        indices = token_id >= self.ori_vocab_size
        indices = (indices * 1).unsqueeze(-1)
        embeds = (1 - indices) * embeds.detach() + indices * embeds
        return embeds
    
    def get_dca_loss(self,shift_logits, shift_labels):
        
        # Flatten the tokens
        
        shift_logits = shift_logits.view(-1, 50257)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.loss_fct(shift_logits, shift_labels)
        return loss

    def forward(self, inputs):
        if inputs["training"]:
            return self.train_forward(inputs)
        else:
            return self.evaluate_forward(inputs)

    def train_forward(self, inputs):
        all_token_id, all_token_mask, tgt_idx, obj_mask = \
            inputs["all_token_id"].int(), inputs["all_token_mask"].int(), inputs["tgt_idx"].int(), inputs["obj_mask"].bool()
        scene_mask3d_feat_pos, scene_mask3d_feat = inputs["scene_mask3d_feat_pos"], inputs["scene_mask3d_feat"]
        scene_feat, scene_img_feat = inputs["scene_feat"], inputs["scene_img_feat"]
        clicks, point_mins, point_maxs = inputs["clicks"], inputs["point_mins"], inputs["point_maxs"]
        click_token_index = inputs["click_token_index"]
        dca_id, dca_mask = inputs["dca_token_id"], inputs["dca_token_mask"]
        batch_size = scene_mask3d_feat.shape[0]
        clicks = clicks.to(scene_feat.dtype)
        point_mins = point_mins.to(scene_feat.dtype)
        point_maxs = point_maxs.to(scene_feat.dtype)
        device = scene_feat.device
        object_embed = torch.cat([F.normalize(scene_mask3d_feat, dim=-1), 
                                    F.normalize(scene_feat, dim=-1)], dim=-1)
        proj_object_embed = self.object_proj(object_embed) + self.object_pos_proj(F.normalize(scene_mask3d_feat_pos))
        
        
        if dca_mask.sum() == 0:
            dca_loss = torch.tensor(0.0).to(device)
        else:
            valid_object = dca_mask.sum(-1) != 0 
            gt2_input_embeds = torch.cat([proj_object_embed.unsqueeze(2), self.transformer.transformer.wte(dca_id)], dim=2)
            gt2_inputs_masks = torch.cat([torch.ones_like(proj_object_embed[...,0].unsqueeze(-1)), dca_mask], dim=2) 

            with self.maybe_autocast():
                outputs = self.transformer( # num_annotated x (1 + max_des_len)
                    inputs_embeds=gt2_input_embeds[valid_object],
                    attention_mask=gt2_inputs_masks[valid_object],
                    encoder_hidden_states=None,
                )

            dca_loss = self.get_dca_loss(outputs.logits[..., :-1, :].contiguous(),dca_id[valid_object].contiguous())
        
        
        proj_object_img_embed = self.object_img_proj(F.normalize(scene_img_feat, dim=-1))
    
        n_obj, n_embed = proj_object_embed.shape[1:3]
        objid_embeds = self.get_objid_embeds()
        objid_embeds = objid_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

        num_expanded = 3 + 1
        object_list_embed = torch.zeros((batch_size, n_obj * num_expanded, n_embed), 
                                        dtype=objid_embeds.dtype, device=objid_embeds.device)
    
        object_list_embed[:, 0:n_obj * num_expanded:num_expanded, :] = objid_embeds
        object_list_embed[:, 1:n_obj * num_expanded:num_expanded, :] = proj_object_embed 
        object_list_embed[:, 2:n_obj * num_expanded:num_expanded, :] = proj_object_img_embed 
        object_list_embed[:, 3:n_obj * num_expanded:num_expanded, :] = objid_embeds

        input_embed = self.get_text_emb(all_token_id)
        input_embed[obj_mask] = object_list_embed.view(-1, n_embed)

        if inputs["training"]:
            max_len = 976
            if input_embed.shape[1] > max_len:
                input_embed = torch.cat([input_embed[:, :max_len-1], input_embed[:, -1:]], dim=1)
                all_token_mask = torch.cat([all_token_mask[:, :max_len-1], all_token_mask[:, -1:]], dim=1)
                tgt_idx = torch.cat([tgt_idx[:, :max_len-1], tgt_idx[:, -1:]], dim=1)
                
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=input_embed,
                    attention_mask=all_token_mask,
                    return_dict=True,
                    labels=tgt_idx.to(torch.int64),
                )
            
            return dict(
                loss=outputs.loss*3+dca_loss, 
                obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
                obj_img_norm=proj_object_img_embed.norm(dim=-1).mean().detach().cpu(),
                objid_norm=objid_embeds.norm(dim=-1).mean().detach().cpu(),
            )
        
        else:
            output_texts = []
            for i in range(batch_size):
                with self.maybe_autocast():
                    outputs = self.llama_model.generate(
                        inputs_embeds=input_embed[[i]][:, :int(sum(all_token_mask[i]))],
                        max_new_tokens=self.max_txt_len,
                        num_beams=5,
                        min_length=1,
                        repetition_penalty=3.0,
                        length_penalty=1,
                        temperature=1.0,
                    )
                output_token = outputs[0]
                output_text = self.llama_tokenizer.decode(output_token)
                output_text = output_text.split(self.end_sym)[0]
                output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
                output_texts.append(output_text)
            return output_texts

    def evaluate_forward(self, inputs):
        all_token_id, all_token_mask, tgt_idx, obj_mask = \
            inputs["all_token_id"].int(), inputs["all_token_mask"].int(), inputs["tgt_idx"].int(), inputs["obj_mask"].bool()
        scene_mask3d_feat_pos, scene_mask3d_feat = inputs["scene_mask3d_feat_pos"], inputs["scene_mask3d_feat"]
        scene_feat, scene_img_feat = inputs["scene_feat"], inputs["scene_img_feat"]
        clicks, point_mins, point_maxs = inputs["clicks"], inputs["point_mins"], inputs["point_maxs"]
        click_token_index = inputs["click_token_index"]

        batch_size = scene_mask3d_feat.shape[0]
        clicks = clicks.to(scene_feat.dtype)
        point_mins = point_mins.to(scene_feat.dtype)
        point_maxs = point_maxs.to(scene_feat.dtype)
        device = scene_feat.device

        object_embed = torch.cat([F.normalize(scene_mask3d_feat, dim=-1), 
                                  F.normalize(scene_feat, dim=-1)], dim=-1)
        proj_object_embed = self.object_proj(object_embed) + self.object_pos_proj(F.normalize(scene_mask3d_feat_pos))
        # proj_object_embed = self.object_proj(F.normalize(scene_feat, dim=-1))
        proj_object_img_embed = self.object_img_proj(F.normalize(scene_img_feat, dim=-1))
       
        n_obj, n_embed = proj_object_embed.shape[1:3]
        objid_embeds = self.get_objid_embeds()
        objid_embeds = objid_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

        num_expanded = 3 + 1
        object_list_embed = torch.zeros((batch_size, n_obj * num_expanded, n_embed), 
                                        dtype=objid_embeds.dtype, device=objid_embeds.device)
       
        object_list_embed[:, 0:n_obj * num_expanded:num_expanded, :] = objid_embeds
        object_list_embed[:, 1:n_obj * num_expanded:num_expanded, :] = proj_object_embed 
        object_list_embed[:, 2:n_obj * num_expanded:num_expanded, :] = proj_object_img_embed 
        object_list_embed[:, 3:n_obj * num_expanded:num_expanded, :] = objid_embeds

        input_embed = self.get_text_emb(all_token_id)
        input_embed[obj_mask] = object_list_embed.view(-1, n_embed)
        output_texts = []
        for i in range(batch_size):
            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=input_embed[[i]][:, :int(sum(all_token_mask[i]))],
                    max_new_tokens=self.max_txt_len,
                    num_beams=5,
                    min_length=1,
                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0,
                )
            output_token = outputs[0]
            output_text = self.llama_tokenizer.decode(output_token)
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
            output_texts.append(output_text)
        return output_texts
    
    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
