"""Resumable full-backbone joint Gemma-270M training on cloud CUDA.

Only train.jsonl and development.jsonl are read. FP32 master parameters and fused
AdamW states stay separate from BF16 inference exports. Development selection
scores the actual exported encoder with a FP32 scalar head, matching serving.
"""
from collections import Counter, defaultdict, deque
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file, save_file
from common import save, schema_state, MODEL_REVISION

SEED=2704206
PRIMARY_FAMILIES=("intent_routing","evidence_relation")
ROOT=Path(os.environ.get("V4_ROOT","/experiment/work/decision-v4"))
DATA=Path(os.environ.get("V4_DATA_ROOT",str(ROOT/"data")))
BASE=Path(os.environ.get("GEMMA_MODEL_PATH","/experiment/work/model"))
INITIAL_HEAD=Path(os.environ.get("V4_INITIAL_HEAD","/experiment/work/decision-v3/joint/best/head.safetensors"))
STATE_CAP,CANDIDATE_CAP=2048,768
EFFECTIVE_GROUPS,MAX_MICRO_GROUPS,MAX_PADDED_TOKENS=16,4,12000
HEAD_CONFIG={"hidden":640,"width":512}
SEPARATOR="\n\nCandidate action:\n"


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_head():return nn.Sequential(nn.LayerNorm(640),nn.Linear(640,512),nn.GELU(),nn.Linear(512,1))


def joint_text(state,candidate):return state+SEPARATOR+candidate


def target(row):
    if 'target_probs' in row:value=np.asarray(row['target_probs'],dtype=np.float64)
    else:
        value=np.zeros(len(row['candidates']),dtype=np.float64);value[int(row['target'])]=1.
    if len(value)!=len(row['candidates']) or not np.isfinite(value).all() or (value<0).any() or abs(value.sum()-1)>1e-5:
        raise ValueError('Invalid target distribution for '+row['id'])
    return value


def _read(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]


def _schedule(rows,epochs,max_steps,seed=SEED):
    """Each epoch visits every eligible row; update groups are distinct prompts."""
    schedule=[];rng=random.Random(seed)
    for epoch in range(epochs):
        order=list(range(len(rows)));rng.shuffle(order);ordered=[]
        for offset in range(0,len(order),256):
            ordered.extend(sorted(order[offset:offset+256],key=lambda i:(rows[i]['maximum_joint_tokens'],rows[i]['id'])))
        batches=[];ordered=deque(ordered)
        while ordered:
            chosen=[];groups=set();deferred=[]
            while ordered and len(chosen)<EFFECTIVE_GROUPS:
                index=ordered.popleft()
                if rows[index]['group'] not in groups:
                    chosen.append(index);groups.add(rows[index]['group'])
                else:deferred.append(index)
            ordered.extendleft(reversed(deferred));batches.append(chosen)
        rng.shuffle(batches)
        schedule.extend({'epoch':epoch,'indices':batch} for batch in batches)
    return schedule[:max_steps]


def _prepare(tokenizer,epochs,max_steps,output,deadline,seed=SEED):
    paths={name:DATA/f'{name}.jsonl' for name in ['train','development']}
    audit_path=DATA/'audit.json'
    audit=json.loads(audit_path.read_text())
    if not audit.get('passed') or not audit.get('passed_structural') or any(
        audit.get(key)!=0 for key in ['prior_holdout_training_conflicts','new_holdout_prior_exposure_conflicts']
    ):raise RuntimeError('The independently prepared v4 data audit has not passed')
    data_hashes={name:_sha(path) for name,path in paths.items()}
    if any(audit['split_file_sha256'].get(f'{name}.jsonl')!=value for name,value in data_hashes.items()):
        raise RuntimeError('Training/development data changed after the v4 audit')
    rows={name:_read(path) for name,path in paths.items()}
    for name,values in rows.items():
        if len({row['id'] for row in values})!=len(values):raise RuntimeError('Duplicate IDs in '+name)
    if {row['group'] for row in rows['train']} & {row['group'] for row in rows['development']}:
        raise RuntimeError('Training and development prompt groups overlap')
    token_cache={};combined=[];combined_index={};excluded=Counter()
    def ids(text):
        if text not in token_cache:token_cache[text]=tokenizer(text,add_special_tokens=True)['input_ids']
        return token_cache[text]
    print(f"V4 preparing {len(rows['train'])} training / {len(rows['development'])} development rows",flush=True)
    for split,values in rows.items():
        for i,row in enumerate(values):
            if i%1000==0:
                print(f'V4 token preparation {split} {i}/{len(values)}',flush=True)
                if time.monotonic()>deadline-180:raise TimeoutError('Insufficient allocation after data preparation')
            if not 2<=len(row['candidates'])<=64 or len(set(row['candidates']))!=len(row['candidates']):raise ValueError('Invalid candidate set')
            target(row);rendered=schema_state(row['state'],row.get('question',''))
            lengths=[len(ids(rendered)),*[len(ids(candidate)) for candidate in row['candidates']]]
            row['eligible']=lengths[0]<=STATE_CAP and max(lengths[1:])<=CANDIDATE_CAP
            row['joint_indices']=[]
            if not row['eligible']:
                if split=='train':excluded[row['source']]+=1
                row['maximum_joint_tokens']=0;continue
            for candidate in row['candidates']:
                text=joint_text(rendered,candidate)
                if text not in combined_index:
                    combined_index[text]=len(combined);combined.append(ids(text))
                row['joint_indices'].append(combined_index[text])
            row['maximum_joint_tokens']=max(len(combined[index]) for index in row['joint_indices'])
    training=[row for row in rows['train'] if row['eligible']];development=rows['development']
    if not training or any(not any(row['family']==family for row in development) for family in PRIMARY_FAMILIES):
        raise RuntimeError('Missing eligible training or a primary development family')
    schedule=_schedule(training,epochs,max_steps,seed)
    manifest={'format':'gemmadecision-full-joint-v4','seed':seed,'run_name':output.name,'base_revision':MODEL_REVISION,
        'base_weights_sha256':_sha(BASE/'model.safetensors'),'initial_head_sha256':_sha(INITIAL_HEAD),
        'initialization':'Official Gemma encoder + selected v3 joint scalar head; all parameters become trainable',
        'data_sha256':data_hashes,'passed_data_audit_sha256':_sha(audit_path),'trainer_source_sha256':_sha(__file__),
        'development_ids':[row['id'] for row in development],'development_ids_sha256':_digest([row['id'] for row in development]),
        'training_rows':len(training),'development_rows':len(development),'training_length_exclusions':dict(excluded),
        'training_family_counts':dict(Counter(row['family'] for row in training)),
        'epochs':epochs,'maximum_steps':max_steps,'planned_steps':len(schedule),
        'schedule_sha256':_digest([[training[index]['id'] for index in batch['indices']] for batch in schedule]),
        'scheduled_examples':sum(len(batch['indices']) for batch in schedule),
        'primary_development_families':list(PRIMARY_FAMILIES),'selection':'Equal primary-family categorical accuracy; NLL tie-breaker. Synthetic/replay families are diagnostic only.',
        'controls':{'effective_groups':EFFECTIVE_GROUPS,'max_micro_groups':MAX_MICRO_GROUPS,'max_padded_tokens_per_forward':MAX_PADDED_TOKENS,
            'encoder_lr':2e-5,'head_lr':2e-4,'weight_decay':.01,'warmup_fraction':.05,'scheduler':'linear warmup then cosine to zero',
            'optimizer':'fused AdamW','master_dtype':'float32','training_autocast':'bfloat16','gradient_clip':1.,
            'validation_every_updates':250,'early_stop_stale_checks':4,'state_limit':STATE_CAP,'candidate_limit':CANDIDATE_CAP,
            'training_loss':'Mean per-prompt candidate soft-target cross entropy','head_config':HEAD_CONFIG},
        'export_contract':{'encoder_dtype':'bfloat16','head_dtype':'float32','autocast_during_selection':False,
                           'joint_separator':SEPARATOR,'pooling':'L2-normalized last non-padding token'},
        'calibration_and_final_opened':False}
    existing=output/'manifest.json'
    if existing.exists() and json.loads(existing.read_text())!=manifest:raise RuntimeError('V4 resume manifest changed')
    save(existing,manifest)
    offsets=np.zeros(len(combined)+1,dtype=np.int64)
    for i,tokens in enumerate(combined):offsets[i+1]=offsets[i]+len(tokens)
    tokens=np.empty(int(offsets[-1]),dtype=np.uint32)
    for i,values in enumerate(combined):tokens[offsets[i]:offsets[i+1]]=values
    del combined,combined_index,token_cache
    print(f"V4 plan: {len(schedule)} updates, {sum(len(x['indices']) for x in schedule)} exposures, {len(tokens)} cached input tokens",flush=True)
    return training,development,schedule,manifest,tokens,offsets


def _microbatches(rows,offsets):
    pending=[]
    def padded(values):
        indices=[i for row in values for i in row['joint_indices']]
        return len(indices)*max(int(offsets[i+1]-offsets[i]) for i in indices)
    for row in rows:
        proposed=pending+[row]
        if pending and (len(proposed)>MAX_MICRO_GROUPS or padded(proposed)>MAX_PADDED_TOKENS):
            yield pending;pending=[row]
        else:pending=proposed
    if pending:yield pending


def _forward(encoder,head,rows,tokens,offsets,pad_id,autocast):
    indices=[i for row in rows for i in row['joint_indices']];scores=[];position=0
    while position<len(indices):
        chosen=[];maximum=0
        for index in indices[position:]:
            length=int(offsets[index+1]-offsets[index]);width=max(maximum,length)
            if chosen and (len(chosen)+1)*width>MAX_PADDED_TOKENS:break
            chosen.append(index);maximum=width
        values=np.full((len(chosen),maximum),pad_id,dtype=np.int64);mask=np.zeros_like(values)
        for j,index in enumerate(chosen):
            sequence=tokens[offsets[index]:offsets[index+1]];values[j,:len(sequence)]=sequence;mask[j,:len(sequence)]=1
        input_ids=torch.tensor(values,device='cuda');attention=torch.tensor(mask,device='cuda')
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=autocast):
            hidden=encoder(input_ids=input_ids,attention_mask=attention,use_cache=False).last_hidden_state
            last=attention.sum(-1)-1;vectors=F.normalize(hidden[torch.arange(len(chosen),device='cuda'),last].float(),dim=-1)
            scores.append(head(vectors).squeeze(-1).float())
        position+=len(chosen)
    flat=torch.cat(scores);result=[];at=0
    for row in rows:
        size=len(row['joint_indices']);result.append(flat[at:at+size]);at+=size
    return result


def _loss(values,rows):
    if len(values)!=len(rows) or not rows:raise ValueError('Loss must score every nonempty batch row')
    return torch.stack([-(torch.tensor(target(row),device=score.device,dtype=torch.float32)*F.log_softmax(score.float(),dim=-1)).sum()
                        for score,row in zip(values,rows)]).mean()


def summarize(rows,values):
    if len(rows)!=len(values) or not rows:raise ValueError('Development scores must match every row')
    grouped=defaultdict(list)
    for row,z in zip(rows,values):
        wanted=target(row);item={'covered':z is not None,'credit':0.,'nll':None,'ordinal_mae':None,
                              'categorical':row.get('metric') not in {'ordinal','preference_tie'}}
        if z is not None:
            z=np.asarray(z,dtype=np.float64)
            if z.shape!=wanted.shape:raise ValueError('Development candidate score shape changed')
            if not np.isfinite(z).all():raise RuntimeError('Non-finite development scores')
            z-=z.max();logp=z-np.log(np.exp(z).sum());p=np.exp(logp)
            winners=np.flatnonzero(z==z.max())
            item.update(credit=float(wanted[winners].mean()),nll=float(-(wanted*logp).sum()))
            if row.get('metric')=='ordinal':
                ratings=np.asarray(row.get('candidate_values',list(range(len(p)))))
                item['ordinal_mae']=float(abs((p*ratings).sum()-row.get('expected_rating',(wanted*ratings).sum())))
        grouped[row['family']].append(item)
    families={}
    for name,items in grouped.items():
        categorical=[x for x in items if x['categorical']];covered=[x for x in categorical if x['covered']]
        ordinal=[x['ordinal_mae'] for x in items if x['ordinal_mae'] is not None]
        families[name]={'count':len(items),'classification_count':len(categorical),
            'accuracy':float(np.mean([x['credit'] for x in categorical])) if categorical else None,
            'nll_covered':float(np.mean([x['nll'] for x in covered])) if covered else None,
            'coverage':sum(x['covered'] for x in items)/len(items),
            'ordinal_mae_covered':float(np.mean(ordinal)) if ordinal else None}
    if any(name not in families or families[name]['accuracy'] is None or families[name]['nll_covered'] is None for name in PRIMARY_FAMILIES):
        raise RuntimeError('A primary development family has no scored categorical cases')
    return {'primary_macro':float(np.mean([families[name]['accuracy'] for name in PRIMARY_FAMILIES])),
            'primary_nll':float(np.mean([families[name]['nll_covered'] for name in PRIMARY_FAMILIES])),
            'by_family':families,'coverage':sum(z is not None for z in values)/len(values)}


def _head_save(head,path):
    save_file({name:value.detach().cpu().contiguous() for name,value in head.state_dict().items()},str(path))


def _export(encoder,head,tokenizer,path,step):
    if path.exists():shutil.rmtree(path)
    path.mkdir(parents=True)
    state={name:value.detach().to(device='cpu',dtype=torch.bfloat16 if value.is_floating_point() else value.dtype).contiguous()
           for name,value in encoder.state_dict().items()}
    encoder.save_pretrained(path,state_dict=state,safe_serialization=True,max_shard_size='2GB')
    config=json.loads((path/'config.json').read_text());config['dtype']='bfloat16';config.pop('torch_dtype',None)
    save(path/'config.json',config);tokenizer.save_pretrained(path)
    # Keep exact upstream tokenizer bytes; runtime sets padding explicitly.
    for source in BASE.iterdir():
        if source.is_file() and (source.name.startswith('tokenizer') or source.name in {
            'special_tokens_map.json','added_tokens.json','chat_template.jinja'}):
            shutil.copy2(source,path/source.name)
    _head_save(head,path/'joint_head.safetensors')
    save(path/'joint_config.json',{'model_name':'GemmaDecision-270M-FullJoint','format':'gemmadecision-full-joint-v4',
        'head_config':HEAD_CONFIG,'head_file':'joint_head.safetensors','max_state_tokens':STATE_CAP,'max_action_tokens':CANDIDATE_CAP,
        'selected_training_step':step,'encoder_dtype':'bfloat16','head_dtype':'float32','base_revision':MODEL_REVISION,
        'score_semantics':'Uncalibrated scalar ranking score; higher is preferred. Not a probability or confidence estimate.'})


@torch.no_grad()
def _evaluate_export(path,rows,tokens,offsets,pad_id):
    encoder=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    head=make_head().to('cuda').eval().requires_grad_(False);head.load_state_dict(load_file(str(path/'joint_head.safetensors'),device='cuda'))
    values=[]
    for offset in range(0,len(rows),4):
        block=rows[offset:offset+4];eligible=[row for row in block if row['eligible']]
        scored=_forward(encoder,head,eligible,tokens,offsets,pad_id,False) if eligible else []
        iterator=iter(scored);values.extend(next(iterator).cpu().tolist() if row['eligible'] else None for row in block)
    summary=summarize(rows,values)
    del encoder,head;gc.collect();torch.cuda.empty_cache()
    return summary,values


def _validate(encoder,head,tokenizer,output,step,development,tokens,offsets,best,history,stale):
    candidate=output/'candidate-export';_export(encoder,head,tokenizer,candidate,step)
    metrics,values=_evaluate_export(candidate,development,tokens,offsets,tokenizer.pad_token_id)
    history.append({'step':step,'development':metrics})
    improved=best is None or (metrics['primary_macro'],-metrics['primary_nll'])>(best['score'],-best['nll'])
    if improved:
        best={'step':step,'score':metrics['primary_macro'],'nll':metrics['primary_nll']};stale=0
        previous=output/'best-previous'
        if previous.exists():shutil.rmtree(previous)
        if (output/'best').exists():(output/'best').rename(previous)
        candidate.rename(output/'best')
        save(output/'best-predictions.json',values)
        save(output/'best-selection.json',{'best':best,'metrics':metrics,'export_precision_verified':True,
            'encoder_sha256':_sha(output/'best/model.safetensors'),'head_sha256':_sha(output/'best/joint_head.safetensors')})
        if previous.exists():shutil.rmtree(previous)
    else:
        stale+=1;shutil.rmtree(candidate)
    print(f"V4 development step{step}: primary={metrics['primary_macro']:.4f} NLL={metrics['primary_nll']:.4f}; beststep={best['step']} stale={stale}",flush=True)
    return best,stale


def _checkpoint(encoder,head,optimizer,output,step,best,history,stale,manifest,checks):
    old=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else None
    destination=output/f'latest-{step:05d}'
    temporary=output/f'latest-{step:05d}-writing'
    if temporary.exists():shutil.rmtree(temporary)
    temporary.mkdir()
    encoder.save_pretrained(temporary,safe_serialization=True,max_shard_size='2GB')
    _head_save(head,temporary/'joint_head.safetensors')
    torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()},temporary/'optimizer.pt')
    if destination.exists():shutil.rmtree(destination)
    temporary.rename(destination)
    progress={'step':step,'checkpoint':str(destination),'best':best,'history':history,'stale_checks':stale,
              'manifest_sha256':_digest(manifest),'checks':checks}
    save(output/'progress.json',progress)
    if old:
        previous=Path(old['checkpoint'])
        if previous!=destination and previous.parent==output and previous.name.startswith('latest-') and previous.exists():shutil.rmtree(previous)


def run(seconds=2220,max_steps=6000,epochs=4,seed=SEED,run_name='train'):
    if (ROOT/'final-evaluation/final-labels-opened.json').exists():
        raise RuntimeError('Final labels have been opened; further v4 training or development selection is prohibited')
    if not torch.cuda.is_available():raise RuntimeError('Full-backbone training is authorized on cloud CUDA only')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 hardware support is required')
    if not 300<=seconds<=14400 or not 1<=epochs<=4 or not 1<=max_steps<=6000 or run_name not in {'train','train-repeat'}:
        raise ValueError('Outside the bounded v4 training allocation')
    if not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('Seed must be a nonnegative 32-bit integer')
    start=time.monotonic();deadline=start+seconds;torch.set_num_threads(2)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    output=ROOT/run_name;output.mkdir(parents=True,exist_ok=True)
    if not (output/'best').exists() and (output/'best-previous').exists():
        (output/'best-previous').rename(output/'best')
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True);tokenizer.padding_side='right'
    training,development,schedule,manifest,tokens,offsets=_prepare(tokenizer,epochs,max_steps,output,deadline,seed)
    progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else None
    if progress and progress['manifest_sha256']!=_digest(manifest):raise RuntimeError('Resume controls/data changed')
    path=Path(progress['checkpoint']) if progress else BASE
    encoder=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('cuda').train().requires_grad_(True)
    if getattr(encoder.config,'attention_dropout',0.)!=0.:raise RuntimeError('This experiment requires deterministic zero attention dropout')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=make_head().to('cuda').train();head.load_state_dict(load_file(str(path/'joint_head.safetensors' if progress else INITIAL_HEAD),device='cuda'))
    parameters=list(encoder.parameters())+list(head.parameters())
    if not all(p.requires_grad and p.dtype==torch.float32 for p in parameters):raise RuntimeError('Full FP32 trainability contract failed')
    optimizer=torch.optim.AdamW([{'params':list(encoder.parameters()),'lr':2e-5},{'params':list(head.parameters()),'lr':2e-4}],weight_decay=.01,fused=True)
    step=progress['step'] if progress else 0;start_step=step
    best=progress['best'] if progress else None;history=progress['history'] if progress else [];stale=progress['stale_checks'] if progress else 0
    checks=progress['checks'] if progress else {}
    if progress:
        state=torch.load(path/'optimizer.pt',weights_only=True,map_location='cuda');optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']]);del state
    if not history:
        best,stale=_validate(encoder,head,tokenizer,output,0,development,tokens,offsets,best,history,stale)
        _checkpoint(encoder,head,optimizer,output,0,best,history,stale,manifest,checks)
    sample_name,sample=next((name,p) for name,p in encoder.named_parameters() if name.endswith('self_attn.q_proj.weight'))
    sample_before=sample.detach().flatten()[:64].clone();warmup=max(1,math.ceil(.05*len(schedule)))
    last_log=time.monotonic();last_saved=step;losses=[];update_times=[];actual_tokens=0;early_stopped=stale>=4;positive_lr_updates=0
    for index in range(step,len(schedule)):
        if early_stopped or time.monotonic()>deadline-180:break
        encoder.train();head.train();optimizer.zero_grad(set_to_none=True)
        factor=(index+1)/warmup if index<warmup else .5*(1+math.cos(math.pi*(index-warmup)/max(1,len(schedule)-warmup-1)))
        for group,peak in zip(optimizer.param_groups,[2e-5,2e-4]):group['lr']=peak*factor
        batch=[training[i] for i in schedule[index]['indices']];tick=time.monotonic();total_loss=0.
        for micro in _microbatches(batch,offsets):
            values=_forward(encoder,head,micro,tokens,offsets,tokenizer.pad_token_id,True)
            loss=_loss(values,micro)*(len(micro)/len(batch))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();total_loss+=float(loss.detach())
            actual_tokens+=sum(int(offsets[j+1]-offsets[j]) for row in micro for j in row['joint_indices'])
        if 'encoder_gradient_norm' not in checks:
            checks['encoder_gradient_parameter']=sample_name;checks['encoder_gradient_norm']=float(sample.grad.float().norm()) if sample.grad is not None else 0.
            if checks['encoder_gradient_norm']<=0:raise RuntimeError('No encoder gradient')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise RuntimeError('Non-finite gradient norm')
        optimizer.step();step=index+1;losses.append(total_loss);update_times.append(time.monotonic()-tick)
        positive_lr_updates+=int(factor>0)
        if 'optimizer_state_materialized' not in checks:
            adam_states=[state for state in optimizer.state.values() if 'exp_avg' in state]
            checks['optimizer_state_materialized']=len(adam_states)==len(parameters)
            checks['optimizer_moments_float32']=all(state['exp_avg'].dtype==torch.float32 and state['exp_avg_sq'].dtype==torch.float32 for state in adam_states)
            if not checks['optimizer_state_materialized'] or not checks['optimizer_moments_float32']:
                raise RuntimeError('Full FP32 AdamW moment state was not materialized')
        if time.monotonic()-last_log>=20:
            print(f"V4 fulljoint step{step}/{len(schedule)} epoch{schedule[index]['epoch']+1}; loss={np.mean(losses[-20:]):.4f}; {np.mean(update_times[-20:]):.2f}s/update; peak{torch.cuda.max_memory_allocated()/2**30:.2f}GiB",flush=True)
            last_log=time.monotonic()
        if step%250==0 or step==len(schedule):
            optimizer.zero_grad(set_to_none=True)
            best,stale=_validate(encoder,head,tokenizer,output,step,development,tokens,offsets,best,history,stale)
            _checkpoint(encoder,head,optimizer,output,step,best,history,stale,manifest,checks);last_saved=step
            early_stopped=stale>=4
    if step!=last_saved:
        optimizer.zero_grad(set_to_none=True)
        best,stale=_validate(encoder,head,tokenizer,output,step,development,tokens,offsets,best,history,stale)
        _checkpoint(encoder,head,optimizer,output,step,best,history,stale,manifest,checks)
        early_stopped=stale>=4
    checks['encoder_sample_changed_this_invocation']=bool(not torch.equal(sample.detach().flatten()[:64],sample_before)) if step>start_step else None
    if positive_lr_updates and not checks['encoder_sample_changed_this_invocation']:raise RuntimeError('Encoder sample did not update')
    checks['trainable_encoder_parameters']=sum(p.numel() for p in encoder.parameters());checks['trainable_head_parameters']=sum(p.numel() for p in head.parameters())
    checks['original_base_file_unchanged']=_sha(BASE/'model.safetensors')==manifest['base_weights_sha256']
    if not checks['original_base_file_unchanged']:raise RuntimeError('Official base source changed')
    report={'status':'complete' if step==len(schedule) or early_stopped else 'partial','steps':step,'planned_steps':len(schedule),
        'early_stopped':early_stopped,'best':best,'history':history,'checks':checks,'manifest':manifest,
        'selected_export':str(output/'best'),'selected_encoder_sha256':_sha(output/'best/model.safetensors'),
        'selected_head_sha256':_sha(output/'best/joint_head.safetensors'),'selected_backbone_updated':best['step']>0,
        'new_updates':step-start_step,'training_nonpadding_tokens':actual_tokens,
        'mean_update_seconds':float(np.mean(update_times)) if update_times else None,
        'peak_allocated_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-start,
        'calibration_and_final_opened':False,'initialization_head_reused_from_v3':True,
        'limitations':['Checkpoint choice uses development results only; final performance remains unmeasured.',
                       'Routing and evidence determine selection; synthetic challenges and replay are reported separately.',
                       'Inference scores are uncalibrated rankings, not probabilities.']}
    save(output/'metrics.json',report)
    return report
