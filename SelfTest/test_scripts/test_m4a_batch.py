import time
import pandas as pd
from src.batch_processor import HighThroughputOrchestrator

def test_high_throughput_batch():
    print("=== ProtePilot M4A: 工业级高通量并发压力测试 ===")
    
    # 模拟高通量筛选库：5 个具有不同 PTM 组合的单抗候选物
    molecules = [
        {"name": "mAb_A_WildType", "pI": 8.5, "mw": 150.0, "deam_sites": 0, "ox_sites": 0, "gradient_slope": 15.0},
        {"name": "mAb_B_HighDeam", "pI": 8.5, "mw": 150.0, "deam_sites": 3, "ox_sites": 0, "gradient_slope": 15.0},
        {"name": "mAb_C_HighOx",   "pI": 8.5, "mw": 150.0, "deam_sites": 0, "ox_sites": 3, "gradient_slope": 15.0},
        {"name": "mAb_D_Mixed",    "pI": 8.5, "mw": 150.0, "deam_sites": 1, "ox_sites": 1, "gradient_slope": 15.0},
        {"name": "mAb_E_Basic",    "pI": 9.0, "mw": 150.0, "deam_sites": 0, "ox_sites": 0, "gradient_slope": 15.0} # pI 较高
    ]
    
    print(f"\n[任务下发] 准备并行处理 {len(molecules)} 个分子候选物...")
    start_time = time.time()
    
    # 实例化高通量编排器
    orchestrator = HighThroughputOrchestrator()
    
    # 注意：如果 Claude 生成的方法名不是 run_batch，请根据 agents.py 里的实际名字修改这里（如 process_batch 等）
    try:
        results = orchestrator.run_batch(molecules)
    except AttributeError:
        # 兼容性盲猜，如果叫别的名字，请手动修改
        print("⚠️ 找不到 run_batch 方法，请检查 src/batch_processor.py 中执行批量处理的函数名，并修改测试脚本。")
        return

    end_time = time.time()
    
    print("\n" + "="*60)
    print(f" 🚀 高通量并发测试完成！总耗时: {end_time - start_time:.2f} 秒")
    print("="*60)
    
    # 读取 Claude 自动生成的 CSV 汇总报告并打印
    try:
        df = pd.read_csv("batch_summary.csv")
        print("\n[数据洞察] 批量筛选汇总报告 (batch_summary.csv):")
        # 打印核心列，对比各分子的表现
        print(df.to_string(index=False))
    except FileNotFoundError:
        print("\n⚠️ 未找到 batch_summary.csv，可能 Claude 命名成了其他文件，请查看项目目录。")

if __name__ == "__main__":
    test_high_throughput_batch()