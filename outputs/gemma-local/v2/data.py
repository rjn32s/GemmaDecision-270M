"""CPU preparation: real negative alternatives and task-grouped agent steps."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random
import time
import urllib.request
from common import SEED, FORMAT, MAX_STATE, MAX_ACTION, save, sha, norm, read_jsonl, split_group, schema_state

SOURCES = {
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"),
    "commonsense": ("tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155"),
    "adp": ("neulab/agent-data-collection", "31a76bfb0124d77ae7322eabbb0171bf11ee2c67"),
    "lite": ("Lite-Coder/LiteCoder-Terminal-SFT", "6acdbbdb29979e4b8ea717b12accc8214606d087"),
    "when2call": ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53"),
}


def fetch(root, source, filename):
    path = Path(root)/"downloads"/source/filename
    if path.exists(): return path
    path.parent.mkdir(parents=True, exist_ok=True)
    repo, revision = SOURCES[source]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    tmp = path.with_suffix(path.suffix+".part")
    print(f"Downloading {source}/{filename}", flush=True)
    for attempt in range(3):
        try:
            request=urllib.request.Request(url+("?download=true" if attempt else ""),headers={"User-Agent":"GemmaDecision-research/2"})
            with urllib.request.urlopen(request, timeout=120) as r, tmp.open("wb") as out:
                while block := r.read(4*1024*1024): out.write(block)
            tmp.replace(path)
            break
        except Exception as exc:
            print(f"Download attempt {attempt+1} failed for {source}/{filename}: {type(exc).__name__}",flush=True)
            if attempt==2:raise RuntimeError(f"Download failed for {source}/{filename}: {type(exc).__name__}") from None
            time.sleep(2*(attempt+1))
    return path


def action_text(event):
    kind = event["class_"]
    if kind == "api_action":
        return event["function"] + "(" + ", ".join(f"{k}={v}" for k,v in sorted(event["kwargs"].items())) + ")"
    if kind == "code_action": return event.get("language", "code") + ":\n" + event["content"]
    return event.get("content", "")


def agent_steps(row, environment):
    """No action description, subsequent observation or terminal reward in state."""
    events = row["content"]
    # These sources prepend a complete worked demonstration, then reset the task.
    # Its goal/actions must not be treated as the live trajectory or group key.
    reset_markers = {"os":"Now, I will start a new problem", "kg":"A new question:"}
    marker = reset_markers.get(environment)
    if marker:
        resets=[i for i,e in enumerate(events) if e["class_"]=="text_observation" and marker in e.get("content", "")]
        if not resets: return
        events=events[resets[-1]:]
    obs = [e.get("content", "") for e in events if e["class_"] == "text_observation"]
    # AgentInstruct starts with generic setup + acknowledgement + actual task.
    first_action = next((i for i,e in enumerate(events) if e["class_"] in {"api_action","code_action"}), len(events))
    before = [e["content"] for e in events[:first_action] if e["class_"] == "text_observation" and e.get("source") != "agent"]
    if not before: return
    goal = before[-1]
    initial_index = max(i for i,e in enumerate(events[:first_action]) if e["class_"] == "text_observation")
    history = []
    for i,event in enumerate(events[initial_index+1:], start=initial_index+1):
        kind = event["class_"]
        if kind in {"api_action", "code_action", "message_action"}:
            action = action_text(event)
            if action.strip() and not action.strip().lower() in {"ok.", "ok"}:
                yield {"id":f"{environment}:{row['id']}:{i}", "source":"agent_"+environment,
                       "kind":"agent", "group":sha(environment+":"+norm(goal)),
                       "goal":goal, "history":history[-6:].copy(), "positive":action,
                       "success_provenance":"AgentInstruct curated demonstration; outcome not re-executed"}
            history.append("Action: " + action)
        elif kind == "text_observation": history.append("Observation: " + event.get("content", ""))


def lite_steps(row):
    conv = row["conversations"]
    if not conv: return
    initial = conv[0]["value"]
    goal = initial.split("Task Description:",1)[-1].split("Current terminal state:",1)[0].strip()
    parsed = {}
    for i,m in enumerate(conv):
        if m["from"] != "gpt": continue
        try: parsed[i] = json.loads(m["value"])
        except (ValueError, TypeError): pass
    # Explicitly filter for a demonstrated completion flag. This is self-reported,
    # not a claim that an independent environment verifier was rerun.
    if not any(isinstance(v,dict) and v.get("task_complete") is True for v in parsed.values()): return
    history = []
    for i,m in enumerate(conv[1:],start=1):
        if m["from"] == "gpt":
            data = parsed.get(i)
            if not isinstance(data, dict): continue
            commands = data.get("commands", [])
            if not isinstance(commands,list): continue
            action = "\n".join(str(c.get("keystrokes", "")) for c in commands if isinstance(c,dict)).strip()
            if not action and data.get("task_complete"): action = "Finish: the task is complete."
            if not action: continue
            yield {"id":f"lite:{row['id']}:{i}", "source":"agent_terminal", "kind":"agent",
                   "group":sha("terminal:"+norm(goal)), "goal":goal,"history":history[-6:].copy(),
                   "positive":"bash:\n"+action, "success_provenance":"terminal completion flag; self-reported, not independently verified"}
            history.append("Action: " + action)
        else: history.append("Observation: " + m["value"])


def prepare(root, old_root, model_path):
    import numpy as np
    import pyarrow.parquet as pq
    import ijson
    from transformers import AutoTokenizer
    root, old_root = Path(root), Path(old_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"audit.json").exists():
        existing=json.loads((root/"audit.json").read_text())
        if existing.get("preparation_revision")==3 and existing.get("agent_gate_passed"): return existing
    started=time.monotonic(); counts=Counter(); rows=[]
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)

    @lru_cache(maxsize=256)
    def encoded(value):
        return tuple(tok(value,add_special_tokens=False)["input_ids"])

    instruction="Choose the next action to complete the task."
    overhead=len(encoded("\n\nRecent history:\n\n\n"+instruction))+32

    def add(row):
        if row.get("kind") == "agent":
            goal=row.pop("goal"); history=row.pop("history")
            # Encode each repeated goal/event once. Never tokenize an unbounded
            # concatenation of discarded earlier terminal outputs.
            if len(goal)>12000 or len(encoded(goal))>1024:
                counts[row["source"]+"/goal_overlength"]+=1; return
            budget=MAX_STATE-len(encoded(goal))-overhead
            kept=[]
            for event in reversed(history):
                value=event[-16000:]
                if len(value)!=len(event):counts[row["source"]+"/long_event_character_tail_windowed"]+=1
                ids=encoded(value)
                if len(ids)+2>budget:
                    if not kept:
                        kept.append(tok.decode(ids[-max(1,budget-2):]))
                        counts[row["source"]+"/latest_event_token_tail_windowed"]+=1
                    break
                kept.append(value);budget-=len(ids)+2
            counts[row["source"]+"/old_history_dropped"]+=max(0,len(history)-len(kept))
            row["state"]=schema_state(goal+"\n\nRecent history:\n"+"\n".join(reversed(kept)),instruction)
        row.setdefault("group",sha(norm(row["state"])))
        row.setdefault("split",split_group(row["group"]))
        row.setdefault("options",[row["positive"]]); row.setdefault("target",row["options"].index(row["positive"]))
        if not row["positive"].strip() or len(set(map(norm,row["options"]))) != len(row["options"]):
            counts[row["source"]+"/invalid"]+=1;return
        rows.append(row)

    old_texts=[json.loads(line) for line in (old_root/"texts.jsonl").open()]
    for split in ["train","validation","calibration","test"]:
        for r in read_jsonl(old_root/f"{split}.jsonl"):
            if r["source"] not in {"paq","when2call"}:continue
            old=old_texts[r["s"]]
            context,question=old.removeprefix("State:\n").rsplit("\n\nDecision:\n",1)
            options=[old_texts[r["p"]]] + ([old_texts[r["n"]]] if r["n"]>=0 else [])
            add({"id":r["id"],"source":r["source"],"kind":"qa" if r["source"]=="paq" else "hard",
                 "group":r["group"],"split":split,"state":schema_state(context,question),
                 "positive":options[0],"options":options})
    del old_texts
    for source,files in [("arc",["ARC-Easy/train-00000-of-00001.parquet","ARC-Challenge/train-00000-of-00001.parquet"]),
                         ("commonsense",["data/train-00000-of-00001.parquet"])]:
        for filename in files:
            for r in pq.read_table(fetch(root,source,filename)).to_pylist():
                choices=r["choices"]; keys=choices["label"]; options=choices["text"]
                target=keys.index(r["answerKey"])
                add({"id":source+":"+r["id"],"source":source,"kind":"hard","state":r["question"],
                     "positive":options[target],"options":options,"target":target})
    for env in ["webshop","os","db","kg","alfworld"]:
        fetch(root,"adp",f"agenttuning_{env}/LICENSE")
        for r in read_jsonl(fetch(root,"adp",f"agenttuning_{env}/full_std.jsonl")):
            for row in agent_steps(r,env): add(row)
        print(f"Prepared agent source {env}; {len(rows)} total candidates; elapsed {time.monotonic()-started:.0f}s",flush=True)
    litefile=fetch(root,"lite","litecoder-sft.json")
    lite_count=0
    with litefile.open("rb") as f:
        for trajectory_index,r in enumerate(ijson.items(f,"item"),start=1):
            for row in lite_steps(r):
                add(row);lite_count+=1
            if trajectory_index%100==0:
                print(f"Terminal trajectories {trajectory_index}; extracted steps {lite_count}; total candidates {len(rows)}; elapsed {time.monotonic()-started:.0f}s",flush=True)
            if lite_count>=60000:break
    print(f"Prepared {len(rows)} candidate rows, including {lite_count} terminal steps",flush=True)

    # Exact cross-source state duplicates stay together. Prefer existing v1 split
    # identities; if two preserved splits conflict remove the ambiguous group.
    bystate=defaultdict(list)
    for i,r in enumerate(rows):bystate[sha(norm(r["state"]))].append(i)
    conflicts=set()
    for ids in bystate.values():
        splits={rows[i]["split"] for i in ids}
        if len(splits)>1:conflicts.update(ids)
    seen=set(); valid=[]
    for i,r in enumerate(rows):
        key=(sha(norm(r["state"])),tuple(norm(x) for x in r["options"]),r["target"])
        if i in conflicts:counts[r["source"]+"/split_conflict_removed"]+=1;continue
        if key in seen:counts[r["source"]+"/duplicate"]+=1;continue
        seen.add(key);valid.append(r)
    # Tokenize once into disk-backed arrays; action descriptions are plain text.
    texts=sorted({t for r in valid for t in [r["state"],*r["options"]]})
    lookup={t:i for i,t in enumerate(texts)}; tokens=[]
    for start in range(0,len(texts),512):
        tokens.extend(tok(texts[start:start+512],add_special_tokens=True,truncation=False)["input_ids"])
        if start%65536==0:print(f"Tokenized {start}/{len(texts)}",flush=True)
    packed=[]
    for r in valid:
        s=lookup[r["state"]];options=[lookup[t] for t in r["options"]]
        if len(tokens[s])>MAX_STATE or max(len(tokens[i]) for i in options)>MAX_ACTION:
            counts[r["source"]+"/overlength_rejected"]+=1;continue
        packed.append({k:r[k] for k in ["id","source","kind","group","split","target"]}|
                      {"s":s,"p":options[r["target"]],"options":options})
    used=sorted({i for r in packed for i in [r["s"],*r["options"]]});remap={i:j for j,i in enumerate(used)}
    offsets=np.zeros(len(used)+1,dtype=np.int64)
    for j,i in enumerate(used):offsets[j+1]=offsets[j]+len(tokens[i])
    ids=np.lib.format.open_memmap(root/"tokens.npy",mode="w+",dtype=np.uint32,shape=(int(offsets[-1]),))
    with (root/"texts.jsonl").open("w") as f:
        for j,i in enumerate(used):
            ids[offsets[j]:offsets[j+1]]=tokens[i];f.write(json.dumps(texts[i],ensure_ascii=False)+"\n")
    ids.flush();np.save(root/"offsets.npy",offsets)
    np.save(root/"text-keys.npy",np.array([int(sha(norm(texts[i]))[:15],16) for i in used],dtype=np.int64))
    splits=defaultdict(list)
    for r in packed:
        splits[r["split"]].append(r|{"s":remap[r["s"]],"p":remap[r["p"]],"options":[remap[i] for i in r["options"]]})
    for split,rs in splits.items():
        with (root/f"{split}.jsonl").open("w") as f:
            for r in rs:f.write(json.dumps(r)+"\n")
    group_sets={s:{r["group"] for r in rs} for s,rs in splits.items()}
    for a in group_sets:
        for b in group_sets:
            if a!=b:assert not group_sets[a]&group_sets[b],"Group leakage"
    agent_train=[r for r in splits["train"] if r["kind"]=="agent"]
    agent_val=[r for r in splits["validation"] if r["kind"]=="agent"]
    envs={r["source"] for r in agent_train}
    gate=len(agent_train)>=10000 and len(envs)>=3 and len({r["group"] for r in agent_val})>=100 and envs<={r["source"] for r in agent_val}
    audit={"format":FORMAT,"preparation_revision":3,"sources":SOURCES,"counts":dict(counts),"splits":{s:dict(Counter(r["source"] for r in rs)) for s,rs in splits.items()},
           "texts":len(used),"tokens":int(offsets[-1]),"agent_train_steps":len(agent_train),
           "agent_validation_groups":len({r["group"] for r in agent_val}),"agent_gate_passed":gate,"seconds":time.monotonic()-started,
           "limitations":["AgentInstruct is curated demonstration data; terminal completion is self-reported, not an independently verified reward.",
                          "V1 QA/tool splits preserved; new tasks use 80/10/5/5 grouped splits.",
                          "Exact state conflicts removed; exhaustive cross-source semantic deduplication not claimed.",
                          "Agent history keeps the whole goal (rejecting >12000 characters or >1024 tokens) and recent events; each considered event first keeps its last 16000 characters, then a token tail window is applied if needed. Counts recorded."]}
    save(root/"audit.json",audit)
    if not gate:raise RuntimeError("Mandatory agent coverage gate failed: "+json.dumps(audit))
    return audit
