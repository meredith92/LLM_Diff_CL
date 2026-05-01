# Stage 4: Domain B Continual Learning Script (Windows PowerShell)
# 使用方法: .\run_stage4_continual_learning.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Stage 4: Domain B 持续学习" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ============ 配置参数 ============
$BASE_WORK_DIR = "work_dirs/continual_experiment"
$DOMAIN_A_WORK_DIR = "$BASE_WORK_DIR/stage2_train_domain_a"
$DOMAIN_B_WORK_DIR = "$BASE_WORK_DIR/stage4_continual_finetune_domain_b"
$CONFIG_FILE = "projects/pcb_conductor/configs/segformer_mt_vb.py"

# 持续学习参数
$MAX_ITERS = 1000          # Domain B微调迭代数
$VAL_INTERVAL = 200        # 验证间隔
$USE_LWF = "True"          # 启用Learning Without Forgetting
$LAM_LWF = 1.0             # LWF权重
$BATCH_SIZE = 2
$LR = "6e-5"

Write-Host "配置信息:" -ForegroundColor Yellow
Write-Host "  - Domain A 工作目录: $DOMAIN_A_WORK_DIR"
Write-Host "  - Domain B 工作目录: $DOMAIN_B_WORK_DIR"
Write-Host "  - 配置文件: $CONFIG_FILE"
Write-Host "  - 最大迭代数: $MAX_ITERS"
Write-Host "  - 使用 LWF: $USE_LWF"
Write-Host "  - LWF 权重: $LAM_LWF"
Write-Host ""

# ============ 检查前置条件 ============
Write-Host "检查前置条件..." -ForegroundColor Yellow

if (-not (Test-Path $DOMAIN_A_WORK_DIR)) {
    Write-Host "❌ 错误: Domain A 工作目录不存在!" -ForegroundColor Red
    Write-Host "   $DOMAIN_A_WORK_DIR"
    exit 1
}

Write-Host "✅ Domain A 工作目录存在" -ForegroundColor Green

# 检查是否有Domain A的checkpoint
$DOMAIN_A_CKPT = Get-ChildItem -Path $DOMAIN_A_WORK_DIR -Filter "*.pth" | Select-Object -First 1

if ($DOMAIN_A_CKPT -eq $null) {
    Write-Host "⚠️  警告: 未找到Domain A的checkpoint" -ForegroundColor Yellow
    Write-Host "   将从头开始训练Domain B"
} else {
    Write-Host "✅ 找到Domain A checkpoint: $($DOMAIN_A_CKPT.FullName)" -ForegroundColor Green
}

if (-not (Test-Path $CONFIG_FILE)) {
    Write-Host "❌ 错误: 配置文件不存在!" -ForegroundColor Red
    Write-Host "   $CONFIG_FILE"
    exit 1
}

Write-Host "✅ 配置文件存在" -ForegroundColor Green
Write-Host ""

# ============ 创建工作目录 ============
if (-not (Test-Path $DOMAIN_B_WORK_DIR)) {
    New-Item -ItemType Directory -Path $DOMAIN_B_WORK_DIR -Force | Out-Null
}
Write-Host "✅ 工作目录已就绪: $DOMAIN_B_WORK_DIR" -ForegroundColor Green
Write-Host ""

# ============ 运行训练 ============
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "开始 Domain B 持续学习训练" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$cmd = @(
    "python tools/train.py",
    "--config $CONFIG_FILE",
    "--work-dir $DOMAIN_B_WORK_DIR",
    "--resume",
    "--cfg-options",
    "train_cfg.max_iters=$MAX_ITERS",
    "train_cfg.val_interval=$VAL_INTERVAL",
    "model.use_lwf=$USE_LWF",
    "model.lam_lwf=$LAM_LWF",
    "model.batch_size=$BATCH_SIZE",
    "optim_wrapper.optimizer.lr=$LR",
    "--amp"
) -join " "

Write-Host "执行命令:" -ForegroundColor Yellow
Write-Host $cmd
Write-Host ""

# 执行命令
Invoke-Expression $cmd
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "✅ Stage 4 完成!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "结果保存在: $DOMAIN_B_WORK_DIR"
    Write-Host ""
    Write-Host "下一步: 评估持续学习效果" -ForegroundColor Yellow
    Write-Host "  python projects/pcb_conductor/tools/visualize_results.py"
} else {
    Write-Host ""
    Write-Host "❌ 训练失败!" -ForegroundColor Red
    exit 1
}
