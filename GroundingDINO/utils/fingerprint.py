import os
import json
import hashlib
import datetime
import torch
from typing import Dict, List, Optional, Any


class ModelFingerprint:
    """模型溯源指纹系统
    
    功能：
    1. 绑定权重与提示词
    2. 生成唯一指纹ID
    3. 支持指纹查询和验证
    4. 记录每次训练/推理的完整信息
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.fingerprint_dir = os.path.join(output_dir, "fingerprints")
        os.makedirs(self.fingerprint_dir, exist_ok=True)
        self.fingerprint_index_file = os.path.join(self.fingerprint_dir, "index.json")
        self._load_index()
    
    def _load_index(self):
        if os.path.exists(self.fingerprint_index_file):
            with open(self.fingerprint_index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {"fingerprints": []}
    
    def _save_index(self):
        with open(self.fingerprint_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def generate_params_hash(self, model: torch.nn.Module, sample_size: int = 1000) -> str:
        """生成模型参数哈希"""
        hash_md5 = hashlib.md5()
        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy().tobytes()
                step = max(1, len(param_data) // sample_size)
                hash_md5.update(param_data[::step])
                param_count += 1
        return hash_md5.hexdigest()
    
    def generate_prompts_hash(self, prompts: List[str]) -> str:
        """生成提示词哈希"""
        hash_md5 = hashlib.md5()
        for prompt in sorted(prompts):
            hash_md5.update(prompt.encode('utf-8'))
            hash_md5.update(b'\n')
        return hash_md5.hexdigest()
    
    def create_fingerprint(
        self,
        model: torch.nn.Module,
        prompts: List[str],
        model_type: str = "raw",
        epoch: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建完整的模型指纹
        
        Args:
            model: PyTorch模型
            prompts: 训练/推理使用的提示词列表
            model_type: 模型类型 (raw/fine_tuned/baseline)
            epoch: 训练epoch
            checkpoint_path: 权重文件路径
            metadata: 额外元数据
            
        Returns:
            完整的指纹信息字典
        """
        params_hash = self.generate_params_hash(model)
        prompts_hash = self.generate_prompts_hash(prompts)
        
        fingerprint_id = f"{params_hash[:12]}_{prompts_hash[:12]}"
        
        fingerprint = {
            "fingerprint_id": fingerprint_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_type": model_type,
            "epoch": epoch,
            "params_hash": params_hash,
            "prompts_hash": prompts_hash,
            "prompt_count": len(prompts),
            "prompt_sample": prompts[:10] if prompts else [],
            "checkpoint_path": checkpoint_path,
            "metadata": metadata or {},
            "config": {
                "model_type": model_type,
                "total_prompts": len(prompts),
            }
        }
        
        self._save_fingerprint(fingerprint)
        return fingerprint
    
    def _save_fingerprint(self, fingerprint: Dict[str, Any]):
        """保存指纹到文件"""
        fingerprint_file = os.path.join(
            self.fingerprint_dir,
            f"{fingerprint['fingerprint_id']}.json"
        )
        with open(fingerprint_file, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, indent=2, ensure_ascii=False)
        
        self.index["fingerprints"].append({
            "fingerprint_id": fingerprint["fingerprint_id"],
            "timestamp": fingerprint["timestamp"],
            "model_type": fingerprint["model_type"],
            "epoch": fingerprint["epoch"],
            "file": os.path.basename(fingerprint_file)
        })
        self._save_index()
    
    def verify_fingerprint(
        self,
        model: torch.nn.Module,
        prompts: List[str],
        fingerprint_id: str
    ) -> Dict[str, Any]:
        """验证模型与指纹是否匹配"""
        fingerprint_file = os.path.join(
            self.fingerprint_dir,
            f"{fingerprint_id}.json"
        )
        
        if not os.path.exists(fingerprint_file):
            return {"valid": False, "error": f"Fingerprint {fingerprint_id} not found"}
        
        with open(fingerprint_file, 'r', encoding='utf-8') as f:
            stored = json.load(f)
        
        current_params_hash = self.generate_params_hash(model)
        current_prompts_hash = self.generate_prompts_hash(prompts)
        
        params_match = current_params_hash == stored["params_hash"]
        prompts_match = current_prompts_hash == stored["prompts_hash"]
        
        return {
            "valid": True,
            "params_match": params_match,
            "prompts_match": prompts_match,
            "stored_fingerprint": stored,
            "current_params_hash": current_params_hash,
            "current_prompts_hash": current_prompts_hash
        }
    
    def list_fingerprints(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有指纹"""
        fingerprints = self.index["fingerprints"]
        if model_type:
            fingerprints = [f for f in fingerprints if f["model_type"] == model_type]
        return fingerprints
    
    def get_fingerprint(self, fingerprint_id: str) -> Optional[Dict[str, Any]]:
        """获取指定指纹"""
        fingerprint_file = os.path.join(
            self.fingerprint_dir,
            f"{fingerprint_id}.json"
        )
        if os.path.exists(fingerprint_file):
            with open(fingerprint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def create_comparison_report(
        self,
        fingerprint_ids: List[str],
        results: Dict[str, Any]
    ) -> str:
        """创建对比报告"""
        fingerprints = []
        for fid in fingerprint_ids:
            fp = self.get_fingerprint(fid)
            if fp:
                fingerprints.append(fp)
        
        report = {
            "comparison_timestamp": datetime.datetime.now().isoformat(),
            "fingerprints": fingerprints,
            "results": results,
            "summary": {}
        }
        
        if len(results) >= 2:
            ids = list(results.keys())
            report["summary"] = {
                "improvement_abs_iou": results[ids[1]]["avg_iou"] - results[ids[0]]["avg_iou"] if len(results) >= 2 else 0,
                "improvement_pct": ((results[ids[1]]["avg_iou"] - results[ids[0]]["avg_iou"]) / max(results[ids[0]]["avg_iou"], 0.001)) * 100
            }
        
        report_file = os.path.join(self.fingerprint_dir, "comparison_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_file


def create_baseline_fingerprint(model, prompts, output_dir="./output"):
    """创建裸模型基线指纹"""
    fp_system = ModelFingerprint(output_dir)
    return fp_system.create_fingerprint(
        model=model,
        prompts=prompts,
        model_type="raw",
        metadata={"description": "Baseline: model with random initialization, no fine-tuning"}
    )


def create_finetuned_fingerprint(model, prompts, epoch, checkpoint_path, output_dir="./output"):
    """创建微调模型指纹"""
    fp_system = ModelFingerprint(output_dir)
    return fp_system.create_fingerprint(
        model=model,
        prompts=prompts,
        model_type="fine_tuned",
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        metadata={"description": f"Fine-tuned model at epoch {epoch}"}
    )
