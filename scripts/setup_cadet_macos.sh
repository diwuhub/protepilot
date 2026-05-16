#!/usr/bin/env bash
# =============================================================================
# setup_cadet_macos.sh  ·  PharmaDev AI
# =============================================================================
# 在 macOS 上自动为 ProtePilot 配置 CADET 动态库。
#
# 用法（在 Mac Terminal 里运行，不需要 sudo）:
#   chmod +x ~/ProtePilot/setup_cadet_macos.sh
#   ~/ProtePilot/setup_cadet_macos.sh
#
# 脚本会按顺序尝试以下三种方案，找到第一个可用的即停止:
#   方案 A — 从已有 conda 安装中提取 libcadet.dylib
#   方案 B — 在 Mac 全局搜索 libcadet.dylib（覆盖 brew、旧版安装包等）
#   方案 C — 用 curl 下载官方 binary release 并解压
# =============================================================================

set -euo pipefail

# ── 颜色输出 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
info() { echo -e "${CYAN}[→]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
sep()  { echo -e "${BOLD}──────────────────────────────────────────────────${NC}"; }

# ── 路径配置（根据你的实际情况修改 PROJECT_DIR）────────────────────────────
PROJECT_DIR="${HOME}/ProtePilot"
LIB_DIR="${PROJECT_DIR}/lib"
ENGINE_DIR="${PROJECT_DIR}/engine"
CADET_VERSION="5.1.0"

sep
echo -e "${BOLD}  CADET macOS 动态库安装脚本${NC}"
sep
echo "  项目目录  : ${PROJECT_DIR}"
echo "  库目录    : ${LIB_DIR}"
echo "  引擎目录  : ${ENGINE_DIR}"
echo "  CADET版本 : ${CADET_VERSION}"
sep

# ── 确认项目目录存在 ─────────────────────────────────────────────────────────
if [[ ! -d "${PROJECT_DIR}" ]]; then
  err "项目目录不存在: ${PROJECT_DIR}"
  echo "  请修改脚本顶部的 PROJECT_DIR 变量后重试。"
  exit 1
fi

mkdir -p "${LIB_DIR}"
info "库目录已就绪: ${LIB_DIR}"

# =============================================================================
# 方案 A — 从 conda 环境提取 libcadet
# =============================================================================
try_conda() {
  sep
  echo -e "${BOLD}  方案 A — conda-forge (推荐)${NC}"
  sep

  local conda_cmd=""
  for cmd in conda mamba micromamba; do
    if command -v "${cmd}" &>/dev/null; then
      conda_cmd="${cmd}"
      ok "找到 ${cmd}: $(${cmd} --version 2>&1 | head -1)"
      break
    fi
  done

  if [[ -z "${conda_cmd}" ]]; then
    warn "未找到 conda / mamba，跳过方案 A"
    return 1
  fi

  # 检查是否已安装 cadet-core
  local cadet_lib
  cadet_lib=$(${conda_cmd} run -n base find "${CONDA_PREFIX:-$(${conda_cmd} info --base)/envs/base}" \
    -name "libcadet*.dylib" 2>/dev/null | head -1 || true)

  # 也搜索所有 conda 环境
  if [[ -z "${cadet_lib}" ]]; then
    info "在所有 conda 环境中搜索 libcadet …"
    local base
    base=$(${conda_cmd} info --base 2>/dev/null || echo "")
    if [[ -n "${base}" ]]; then
      cadet_lib=$(find "${base}" -name "libcadet*.dylib" 2>/dev/null | head -1 || true)
    fi
  fi

  if [[ -n "${cadet_lib}" ]]; then
    local lib_source_dir
    lib_source_dir=$(dirname "${cadet_lib}")
    ok "找到已安装的 libcadet: ${cadet_lib}"
    info "复制库文件 …"
    cp "${lib_source_dir}"/libcadet*.dylib "${LIB_DIR}/" 2>/dev/null && ok "复制完成" || true

    # 同时更新 cadet-cli（conda 版本可能更适合）
    local conda_cli
    conda_cli=$(find "$(dirname "${lib_source_dir}")/bin" -name "cadet-cli" 2>/dev/null | head -1 || true)
    if [[ -n "${conda_cli}" ]]; then
      info "更新 engine/cadet-cli (来自 conda) …"
      cp "${conda_cli}" "${ENGINE_DIR}/cadet-cli"
      chmod +x "${ENGINE_DIR}/cadet-cli"
      ok "cadet-cli 已更新"
    fi
    return 0
  fi

  # 未安装，尝试现在安装
  echo ""
  info "未检测到已安装的 cadet-core，尝试通过 conda-forge 安装 …"
  info "命令: ${conda_cmd} install -c conda-forge cadet-core -y"
  echo ""

  if ${conda_cmd} install -c conda-forge cadet-core -y 2>&1; then
    ok "cadet-core 安装成功"
    local base
    base=$(${conda_cmd} info --base 2>/dev/null || echo "")
    cadet_lib=$(find "${base}" -name "libcadet*.dylib" 2>/dev/null | head -1 || true)

    if [[ -n "${cadet_lib}" ]]; then
      cp "$(dirname "${cadet_lib}")"/*.dylib "${LIB_DIR}/" 2>/dev/null
      ok "libcadet.dylib 已复制到 ${LIB_DIR}/"

      # 更新 cadet-cli
      local conda_cli
      conda_cli=$(find "${base}" -name "cadet-cli" -not -path "*/cadet-cli/*" 2>/dev/null | head -1 || true)
      if [[ -n "${conda_cli}" ]]; then
        cp "${conda_cli}" "${ENGINE_DIR}/cadet-cli"
        chmod +x "${ENGINE_DIR}/cadet-cli"
        ok "cadet-cli 已从 conda 更新"
      fi
      return 0
    fi
  fi

  warn "conda 安装失败或未找到库文件，尝试方案 B"
  return 1
}

# =============================================================================
# 方案 B — 全局搜索 libcadet.dylib（覆盖 Homebrew、手动安装包等）
# =============================================================================
try_find() {
  sep
  echo -e "${BOLD}  方案 B — 全局搜索 libcadet.dylib${NC}"
  sep
  info "搜索范围: /usr/local /opt/homebrew ~/Downloads ~/Applications"
  info "（可能需要 10–30 秒）…"

  local found
  found=$(find /usr/local /opt/homebrew "${HOME}/Downloads" "${HOME}/Applications" \
    -name "libcadet*.dylib" 2>/dev/null | head -5 || true)

  if [[ -z "${found}" ]]; then
    warn "未找到 libcadet.dylib，尝试方案 C"
    return 1
  fi

  echo ""
  ok "找到以下库文件:"
  echo "${found}"
  echo ""

  local best
  best=$(echo "${found}" | head -1)
  local src_dir
  src_dir=$(dirname "${best}")

  info "从 ${src_dir} 复制库文件 …"
  cp "${src_dir}"/*.dylib "${LIB_DIR}/" 2>/dev/null && ok "复制完成" || {
    warn "部分文件复制失败，继续…"
  }

  # 寻找配对的 cadet-cli
  local cli_dir
  cli_dir=$(dirname "${src_dir}")/bin
  if [[ -f "${cli_dir}/cadet-cli" ]]; then
    info "找到配套的 cadet-cli，更新 engine/ …"
    cp "${cli_dir}/cadet-cli" "${ENGINE_DIR}/cadet-cli"
    chmod +x "${ENGINE_DIR}/cadet-cli"
    ok "cadet-cli 已更新"
  fi

  return 0
}

# =============================================================================
# 方案 C — 下载官方 binary release
# =============================================================================
try_download() {
  sep
  echo -e "${BOLD}  方案 C — 下载官方 Binary Release${NC}"
  sep

  # 检测 CPU 架构
  local arch
  arch=$(uname -m)
  local release_name

  if [[ "${arch}" == "arm64" ]]; then
    # Apple Silicon
    release_name="cadet-core-v${CADET_VERSION}-macOS-arm64"
  else
    # Intel
    release_name="cadet-core-v${CADET_VERSION}-macOS-x86_64"
  fi

  local url="https://github.com/cadet/CADET-Core/releases/download/v${CADET_VERSION}/${release_name}.tar.gz"
  local tmp_dir
  tmp_dir=$(mktemp -d)
  local tarball="${tmp_dir}/${release_name}.tar.gz"

  info "CPU 架构: ${arch}"
  info "下载地址: ${url}"
  info "下载中（约 20–50 MB）…"

  if ! curl -L --progress-bar -o "${tarball}" "${url}"; then
    warn "下载失败。可能原因:"
    warn "  1. 该架构暂无预编译包（请查看 GitHub Releases 页面确认文件名）"
    warn "  2. 网络问题"
    echo ""
    echo "  请手动下载: https://github.com/cadet/CADET-Core/releases/v${CADET_VERSION}"
    echo "  下载后解压，将 lib/*.dylib 复制到: ${LIB_DIR}/"
    rm -rf "${tmp_dir}"
    return 1
  fi

  info "解压中 …"
  tar -xzf "${tarball}" -C "${tmp_dir}"

  # 找到解压后的 lib 目录
  local lib_src
  lib_src=$(find "${tmp_dir}" -name "libcadet*.dylib" 2>/dev/null | head -1 || true)

  if [[ -z "${lib_src}" ]]; then
    warn "解压包中未找到 libcadet.dylib"
    ls "${tmp_dir}"
    rm -rf "${tmp_dir}"
    return 1
  fi

  local lib_src_dir
  lib_src_dir=$(dirname "${lib_src}")
  info "复制库文件: ${lib_src_dir} → ${LIB_DIR}/"
  cp "${lib_src_dir}"/*.dylib "${LIB_DIR}/"
  ok "库文件复制完成"

  # 更新 cadet-cli
  local cli_src
  cli_src=$(find "${tmp_dir}" -name "cadet-cli" -not -path "*/cadet-cli/*" 2>/dev/null | head -1 || true)
  if [[ -n "${cli_src}" ]]; then
    info "更新 engine/cadet-cli …"
    cp "${cli_src}" "${ENGINE_DIR}/cadet-cli"
    chmod +x "${ENGINE_DIR}/cadet-cli"
    ok "cadet-cli 已更新（来自官方包）"
  fi

  rm -rf "${tmp_dir}"
  return 0
}

# =============================================================================
# 最终验证
# =============================================================================
verify() {
  sep
  echo -e "${BOLD}  最终验证${NC}"
  sep

  local libs
  libs=$(ls "${LIB_DIR}"/*.dylib 2>/dev/null || true)

  if [[ -z "${libs}" ]]; then
    err "lib/ 目录中仍无 .dylib 文件！三种方案均失败。"
    echo ""
    echo "  请手动操作:"
    echo "  1. 前往 https://github.com/cadet/CADET-Core/releases/latest"
    echo "  2. 下载 macOS Binary Release (.tar.gz)"
    echo "  3. 解压后将 lib/*.dylib 复制到: ${LIB_DIR}/"
    exit 1
  fi

  echo "  找到以下库文件:"
  ls -lh "${LIB_DIR}"/*.dylib | awk '{print "    " $NF " (" $5 ")"}'

  echo ""
  info "测试 cadet-cli …"
  local cli="${ENGINE_DIR}/cadet-cli"

  if [[ ! -f "${cli}" ]]; then
    err "cadet-cli 不存在: ${cli}"
    exit 1
  fi

  # 注入 DYLD_LIBRARY_PATH 后试运行
  export DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH:-}"
  local out
  out=$(DYLD_LIBRARY_PATH="${LIB_DIR}" "${cli}" --version 2>&1 || true)

  if echo "${out}" | grep -qi "cadet\|version\|5\.[0-9]"; then
    ok "cadet-cli 运行成功: ${out}"
  else
    # 有时 --version 返回非零但实际是 OK，只要没有 dyld 错误就行
    if echo "${out}" | grep -qi "dyld\|not loaded\|Library"; then
      err "仍有动态库加载错误:"
      echo "  ${out}"
      exit 1
    else
      ok "cadet-cli 可启动（输出: ${out}）"
    fi
  fi

  sep
  echo -e "${GREEN}${BOLD}  ✅ 配置完成！${NC}"
  sep
  echo "  现在可以运行仿真:"
  echo "    cd ${PROJECT_DIR}"
  echo "    python src/cadet_engine.py"
  sep
}

# =============================================================================
# 主流程
# =============================================================================
main() {
  # 按顺序尝试三种方案
  if try_conda 2>/dev/null; then
    :
  elif try_find 2>/dev/null; then
    :
  elif try_download; then
    :
  else
    err "三种方案均失败，请参考上方提示手动操作。"
    exit 1
  fi

  verify
}

main "$@"
