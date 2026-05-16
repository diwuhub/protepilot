"""
verify_engine.py
================
验证 cadet-cli 是否正确放置在 engine/ 目录下。
运行方式: python engine/verify_engine.py
"""

import subprocess
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent

CANDIDATES = [
    ENGINE_DIR / "cadet-cli.exe",   # Windows
    ENGINE_DIR / "cadet-cli",        # Linux / macOS
]


def check():
    found = [p for p in CANDIDATES if p.exists()]

    if not found:
        print("❌ 未找到 cadet-cli 可执行文件！")
        print(f"   请将 cadet-cli（或 cadet-cli.exe）放入：{ENGINE_DIR}")
        print()
        print("   下载地址：https://github.com/cadet/CADET-Core/releases/latest")
        sys.exit(1)

    exe = found[0]
    print(f"✓ 找到可执行文件：{exe}")

    # 运行 --version
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True, text=True, timeout=10
        )
        output = (result.stdout + result.stderr).strip()
        print(f"✓ 版本信息：{output if output else '(无输出，但进程正常退出)'}")
        print()
        print("✅ cadet-cli 就绪，可以开始仿真。")
    except FileNotFoundError:
        print("❌ 文件存在但无法执行（可能缺少执行权限或依赖库）")
        print("   Linux/macOS 请运行: chmod +x engine/cadet-cli")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("⚠️  进程超时，但文件存在。")


if __name__ == "__main__":
    check()
