"""CPU unit tests on tiny RANDOM models, never pretrained research models."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2Config,GPT2LMHeadModel,OPTConfig,OPTForCausalLM,Qwen2Config,Qwen2ForCausalLM
import p4_worker as worker
import p4_io as disk


def main():
    torch.manual_seed(1);torch.set_num_threads(2)
    models={
      'gpt2':GPT2LMHeadModel(GPT2Config(n_embd=32,n_layer=4,n_head=4,n_positions=64,vocab_size=64)),
      'opt':OPTForCausalLM(OPTConfig(hidden_size=32,num_hidden_layers=4,num_attention_heads=4,ffn_dim=64,word_embed_proj_dim=32,vocab_size=64,max_position_embeddings=64,do_layer_norm_before=True)),
      'qwen':Qwen2ForCausalLM(Qwen2Config(hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,vocab_size=64,max_position_embeddings=64))}
    ids=[4,5,6,7,8,4,5,6,7,8,9,10]
    class Tokenizer:
        def __call__(self,text,**kwargs):
            assert kwargs.get('add_special_tokens') is False
            return {'input_ids':torch.tensor([ids],dtype=torch.long)}
        def decode(self,tokens,**kwargs):return str(tokens[0])
        def convert_ids_to_tokens(self,tokens):return list(map(str,tokens))
    old=disk.model;disk.model=lambda key:dict(mid=2,layers=4)
    rows={}
    try:
        for key,m in models.items():
            m.config._attn_implementation='eager';m.eval();m.requires_grad_(False)
            a,meta=worker.extract(m,Tokenizer(),dict(text='synthetic unit fixture',token_ids={key:ids}),key,torch,np,device='cpu')
            checks=meta['extraction_checks'];assert checks['middle_matches'] and checks['final_matches'];assert checks['normalization_max_relative_error']<2e-6;assert checks['surprisal_ce_max_abs']<1e-10
            assert a['h_middle'].shape==a['h_final_norm'].shape==(12,32)
            assert np.isnan(a['surprisal'][0]) and np.isfinite(a['surprisal'][1:]).all()
            rows[key]=checks
    finally:disk.model=old
    result=dict(status='PASS',torch=torch.__version__,transformers='4.57.6',numpy=np.__version__,device='CPU',synthetic_untrained_models=3,pretrained_model_downloads=0,research_model_forward_calls=0,pinned_CUDA_runtime_tested=False,checks=rows)
    disk.write(Path(__file__).with_name('adapter_test_results.json'),result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
