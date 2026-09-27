"""Standalone reproduction entry point; run model stages on a CUDA host."""
import argparse
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--model-path",required=True)
    p.add_argument("--stage",choices=["prepare","assets","cache","reference","train","evaluate"],required=True)
    p.add_argument("--seconds",type=int,default=3300)
    args=p.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    os.environ["GEMMADECISION_WORK_DIR"]=str(root/"decision-v2")
    os.environ["GEMMA_MODEL_PATH"]=str(Path(args.model_path).resolve())
    from common import save,QWEN_REVISION,CLM_REVISION
    if args.stage=="prepare":
        import legacy_data
        from data import prepare
        legacy_data.prepare(root/"decision-v1",args.model_path)
        report=prepare(root/"decision-v2",root/"decision-v1",args.model_path)
    elif args.stage=="assets":
        from huggingface_hub import snapshot_download,hf_hub_download
        snapshot_download("Qwen/Qwen3-8B",revision=QWEN_REVISION,local_dir=root/"qwen-reference",
                          allow_patterns=["*.safetensors","*.json","*.txt","LICENSE"])
        hf_hub_download("Contrastive-LM/CLM-v0.1-8B","CLM_v0.1-8B.pt",revision=CLM_REVISION,local_dir=root/"clm-reference")
        report={"status":"downloaded"}
    else:
        from run import run
        report=run(args.stage,args.seconds)
    save(root/"decision-v2"/args.stage/"metrics.json",report)


if __name__=="__main__":main()
