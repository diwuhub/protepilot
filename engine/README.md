# engine/ — CADET-Core 仿真引擎

此目录用于存放 CADET-Core 可执行文件（`cadet-cli` / `cadet-cli.exe`）。

---

## 下载方式（根据你的操作系统选择）

### Windows（推荐：直接下载安装包）

1. 前往官方 Releases 页面：
   **https://github.com/cadet/CADET-Core/releases/latest**

2. 下载 `cadet-core-*.win64.zip`（或 `.exe` 安装包）

3. 解压后，将 `cadet-cli.exe` 复制到本目录（`engine/`）：
   ```
   ProtePilot/engine/cadet-cli.exe
   ```

---

### Linux x86_64（推荐：conda-forge）

```bash
conda install -c conda-forge cadet-core
# 安装后将可执行文件复制过来
cp $(which cadet-cli) ./engine/cadet-cli
chmod +x ./engine/cadet-cli
```

或手动下载：
1. 前往 **https://github.com/cadet/CADET-Core/releases/latest**
2. 下载 `cadet-core-*.linux.tar.gz`
3. 解压并将 `cadet-cli` 放入本目录

---

### macOS

```bash
conda install -c conda-forge cadet-core
cp $(which cadet-cli) ./engine/cadet-cli
chmod +x ./engine/cadet-cli
```

---

## 验证安装

```bash
# Linux / macOS
./engine/cadet-cli --version

# Windows PowerShell
.\engine\cadet-cli.exe --version
```

预期输出示例：
```
CADET version 5.1.0
```

---

## 目录结构（放置后）

```
engine/
├── cadet-cli          # Linux / macOS 可执行文件
├── cadet-cli.exe      # Windows 可执行文件
└── README.md          # 本说明文件
```

---

## 参考链接

- GitHub 主页：https://github.com/cadet/CADET-Core
- 官方文档：https://cadet.github.io
- conda-forge：https://anaconda.org/conda-forge/cadet-core
