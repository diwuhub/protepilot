import time
import matplotlib.pyplot as plt
from src.agents import PharmaAgentManager
from src.cadet_engine import CadetEngine

def test_global_agent_pipeline():
    print("=== ProtePilot M3: 全局 Agent Workflow 测试 ===")
    
    # 模拟大模型解析用户自然语言后，提取出的结构化意图
    mock_llm_parsed_intent = {
        "name": "mAb_Agent_Test",
        "pI": 8.5,
        "mw": 150.0,
        "deam_sites": 1, # 发现 1 个脱酰胺
        "ox_sites": 1,   # 发现 1 个氧化
        "gradient_slope": 15.0
    }
    
    print(f"\n[Agent Manager] 接收到任务意图: {mock_llm_parsed_intent}")
    
    # 实例化 Agent Manager 并一键运行流水线
    manager = PharmaAgentManager()
    
    try:
        # Agent 内部会自动调用 Bio-Info Tool 和 Simulation Tool
        final_report = manager.run_deterministic_pipeline(mock_llm_parsed_intent)
    except Exception as e:
        print(f"\n❌ Agent 执行失败: {e}")
        return

    # 输出 Agent 整理的最终报告
    print("\n" + "="*50)
    print(" 🤖 Agent 分析完成报告")
    print("="*50)
    
    # 1. 直接打印 Agent 生成的极其专业的 Summary 报告
    print("\n" + final_report.get("summary", "没有找到报告内容"))
            
    # 2. 从深层字典中安全地提取 HDF5 路径以绘制图谱
    h5_path = None
    try:
        # 在 pipeline 的第二个 tool (Simulation) 的返回结果中寻找
        h5_path = final_report['pipeline'][1]['result']['data']['h5_path']
    except (KeyError, IndexError):
        print("\n⚠️ 无法从 Agent 返回结果中解析到 h5_path，跳过画图。")

    if h5_path:
        print(f"\n[UI 渲染] 正在提取 {h5_path} 进行可视化...")
        engine = CadetEngine(workspace="data", engine_dir="engine")
        
        # 具有全局视野的正确底层解包：直接将元组拆分为 时间矩阵 和 浓度字典
        time_array, outlet_dict = engine.read_results(h5_path)[:2]
        
        # time_array 直接就是 numpy 数组，不需要再调 .time
        t_min = time_array / 60.0 
        
        plt.figure(figsize=(10, 6))
        # 从字典中精准提取各组分浓度，而不是调属性
        plt.plot(t_min, outlet_dict['Acidic'], label="Acidic Variant", color="#FF4B4B", linestyle="--", linewidth=2)
        plt.plot(t_min, outlet_dict['Main'], label="Main Peak", color="#1F77B4", linewidth=2.5)
        plt.plot(t_min, outlet_dict['Basic'], label="Basic Variant", color="#2CA02C", linestyle=":", linewidth=2)
        
        plt.title("M3 Agent Workflow Output (Charge Variants)", fontsize=14)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("Concentration (mM)", fontsize=12)
        plt.xlim(0, 50) 
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        import time # 确保有 time 模块
        timestamp = time.strftime("%H%M%S")
        plt.savefig(f"m3_agent_test_{timestamp}.png", dpi=300)
        print(f"✅ 全局测试成功！渲染图已保存为 m3_agent_test_{timestamp}.png")

if __name__ == "__main__":
    test_global_agent_pipeline()