# lib/ — CADET 动态链接库目录

`cadet-cli` 在运行时依赖此目录中的共享库（`.dylib` / `.so`）。

---

## 错误原因

```
dyld: Library not loaded: @rpath/libcadet.0.dylib
  Reason: tried: '../lib/libcadet.0.dylib' (no such file)
```

`cadet-cli` 的 `@rpath` 硬编码指向 `../lib/`（即本目录），
只复制可执行文件而不复制库文件会导致此错误。

---

## 修复方法

从 CADET 官方安装包中将 `lib/` 目录下的所有文件复制到这里。

### macOS（从 .zip 解压后）

```bash
# 假设你的 CADET 安装包解压到了 ~/Downloads/cadet-macos/
cp ~/Downloads/cadet-macos/lib/*.dylib  <project_root>/lib/
```

### Linux

```bash
cp ~/cadet-linux/lib/*.so*  <project_root>/lib/
```

---

## 目录结构（放置后）

```
lib/
├── libcadet.0.dylib       ← 核心库（必须）
├── libcadet.dylib         ← 软链接
├── libhdf5.dylib          ← HDF5 依赖（可能需要）
├── libsuperlu_dist.dylib  ← SuperLU 线性代数库（可能需要）
└── README.md              ← 本说明文件
```

> **注意**：不同 CADET 版本的库文件名称可能略有不同，以实际解压内容为准。
> 通常只需复制整个 `lib/` 目录的内容即可。

---

## 验证

修复后运行：
```bash
python src/cadet_engine.py
```

或直接验证动态库加载：
```bash
# macOS
otool -L engine/cadet-cli

# Linux
ldd engine/cadet-cli
```
