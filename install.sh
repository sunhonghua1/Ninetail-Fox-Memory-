#!/bin/bash
# ============================================================
# OpenClaw Memory V4 一键安装脚本
# 真向量语义搜索 + BM25 混合检索 + 多供应商 Embedding
# ============================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================================"
echo "  🧠 OpenClaw Memory V4 Installer"
echo "  真向量语义搜索 · 多供应商 Embedding · 自动 Fallback"
echo "============================================================"
echo -e "${NC}"

# ─── 检测 OpenClaw 安装路径 ─────────────────────────

OPENCLAW_DIR="$HOME/.openclaw"
SKILL_DIR="$OPENCLAW_DIR/skills/openclaw-memory"
MEMORY_DIR="$OPENCLAW_DIR/memory"

if [ ! -d "$OPENCLAW_DIR" ]; then
    echo -e "${RED}❌ 未检测到 OpenClaw 安装（$OPENCLAW_DIR 不存在）${NC}"
    echo "请先安装 OpenClaw: https://openclaw.ai"
    exit 1
fi

echo -e "${GREEN}✅ 检测到 OpenClaw: $OPENCLAW_DIR${NC}"

# ─── 创建目录 ───────────────────────────────────────

mkdir -p "$SKILL_DIR"
mkdir -p "$MEMORY_DIR"

# ─── 备份旧版本 ──────────────────────────────────────

if [ -f "$SKILL_DIR/openclaw_memory_enhanced.py" ]; then
    BACKUP_NAME="openclaw_memory_enhanced.py.$(date +%Y%m%d_%H%M%S).bak"
    cp "$SKILL_DIR/openclaw_memory_enhanced.py" "$SKILL_DIR/$BACKUP_NAME"
    echo -e "${YELLOW}📦 已备份旧版本: $BACKUP_NAME${NC}"
fi

# ─── 获取脚本所在目录 ─────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── 复制核心文件 ─────────────────────────────────────

echo -e "\n${BLUE}📁 安装核心文件...${NC}"

cp "$SCRIPT_DIR/embedding_provider.py" "$SKILL_DIR/"
echo "  ✅ embedding_provider.py"

cp "$SCRIPT_DIR/openclaw_memory_enhanced.py" "$SKILL_DIR/"
echo "  ✅ openclaw_memory_enhanced.py"

# ─── 配置文件处理 ─────────────────────────────────────

CONFIG_FILE="$SKILL_DIR/embedding_config.json"

if [ -f "$CONFIG_FILE" ]; then
    echo -e "  ${YELLOW}⚠️  embedding_config.json 已存在，跳过（保留你的 API Key）${NC}"
else
    cp "$SCRIPT_DIR/embedding_config.example.json" "$CONFIG_FILE"
    echo "  ✅ embedding_config.json（模板已创建）"
fi

# ─── 配置 API Key ─────────────────────────────────────

echo -e "\n${BLUE}🔑 配置 Embedding API Key${NC}"
echo "  至少需要配置一个供应商的 API Key（推荐 DashScope 或 Jina，都有免费额度）"
echo ""
echo "  供应商          免费额度              获取地址"
echo "  ──────────      ──────────            ────────"
echo "  DashScope       100 万 tokens         https://dashscope.aliyuncs.com"
echo "  Google Gemini   充足                  https://aistudio.google.com"
echo "  Jina AI         1000 万 tokens/月     https://jina.ai/embeddings"
echo ""

# 读取用户输入
read -p "是否现在配置 API Key? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""

    # DashScope
    read -p "DashScope API Key (留空跳过): " DASHSCOPE_KEY
    if [ -n "$DASHSCOPE_KEY" ]; then
        sed -i "s/YOUR_DASHSCOPE_API_KEY/$DASHSCOPE_KEY/g" "$CONFIG_FILE"
        echo -e "  ${GREEN}✅ DashScope 已配置${NC}"
    fi

    # Google
    read -p "Google Gemini API Key (留空跳过): " GOOGLE_KEY
    if [ -n "$GOOGLE_KEY" ]; then
        sed -i "s/YOUR_GOOGLE_API_KEY/$GOOGLE_KEY/g" "$CONFIG_FILE"
        echo -e "  ${GREEN}✅ Google 已配置${NC}"
    fi

    # Jina
    read -p "Jina AI API Key (留空跳过): " JINA_KEY
    if [ -n "$JINA_KEY" ]; then
        sed -i "s/YOUR_JINA_API_KEY/$JINA_KEY/g" "$CONFIG_FILE"
        echo -e "  ${GREEN}✅ Jina 已配置${NC}"
    fi
else
    echo -e "\n${YELLOW}⚠️  请稍后手动编辑: $CONFIG_FILE${NC}"
fi

# ─── 验证安装 ─────────────────────────────────────────

echo -e "\n${BLUE}🧪 验证安装...${NC}"

# 检查 Python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 未找到 python3，请先安装${NC}"
    exit 1
fi

# 检查文件完整性
MISSING=0
for f in embedding_provider.py openclaw_memory_enhanced.py embedding_config.json; do
    if [ -f "$SKILL_DIR/$f" ]; then
        echo -e "  ✅ $f"
    else
        echo -e "  ${RED}❌ $f 缺失${NC}"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo -e "\n${RED}❌ 安装不完整，请检查上述缺失文件${NC}"
    exit 1
fi

# 测试 Embedding（如果有配置 Key）
if grep -q "YOUR_" "$CONFIG_FILE"; then
    echo -e "\n${YELLOW}⚠️  检测到未配置的 API Key，跳过连通性测试${NC}"
    echo "  请编辑 $CONFIG_FILE 填入你的 API Key 后运行:"
    echo "  cd $SKILL_DIR && python3 embedding_provider.py"
else
    echo -e "\n${BLUE}  测试 Embedding API 连通性...${NC}"
    cd "$SKILL_DIR"
    if python3 -c "from embedding_provider import MultiProviderEmbedding; m = MultiProviderEmbedding(config_path='embedding_config.json'); r = m.embed(['test']); print(f'  ✅ {r.provider} 连通成功，维度: {r.dimensions}')" 2>/dev/null; then
        echo -e "  ${GREEN}✅ Embedding API 测试通过${NC}"
    else
        echo -e "  ${YELLOW}⚠️  Embedding API 测试失败，请检查 API Key${NC}"
    fi
fi

# ─── 完成 ──────────────────────────────────────────────

echo -e "\n${GREEN}"
echo "============================================================"
echo "  ✅ 安装完成!"
echo "============================================================"
echo -e "${NC}"
echo "  📁 安装路径: $SKILL_DIR"
echo "  📝 配置文件: $CONFIG_FILE"
echo ""
echo "  下一步:"
echo "  1. 确保 embedding_config.json 中至少有一个 API Key"
echo "  2. 测试: cd $SKILL_DIR && python3 embedding_provider.py"
echo "  3. 重启: openclaw gateway restart"
echo ""
echo "  📖 文档: https://github.com/sunhonghua1/openclaw-memory-v4"
echo ""
