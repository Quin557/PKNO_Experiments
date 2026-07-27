# PKNO 实验准备报告

日期：2026-07-27  
目标：为新建 PKNO/AM-KNO 实验项目准备数据可行性、实验路线、PyTorch 迁移方案、GitHub 仓库结构与 Codex 参考材料清单。

## 1. 当前目标的重新整理

当前不优先写论文叙事，而是先确认实验能不能做、先做什么、代码和数据怎么组织。

你的核心模型方向不是简单优化 KNO，而是逐步走向：

```text
PKNO = Parameterized Koopman Neural Operator
     = shared dictionary / shared observables
     + parameterized Koopman family K(c, u_n)
     + AM-style frequency/kernel generation
     + optional high-frequency refinement
```

其中：

- `shared dictionary` 来自 PKNN/parametric Koopman 的思想：不同参数条件下共享一套观测函数/字典，让状态先进入统一 Koopman observable space。
- `K(c, u_n)` 表示 Koopman operator 不再固定，而由物理条件、当前状态摘要、边界条件或频率条件参数化。
- `AM-style frequency/kernel generation` 来自 AM-FNO 思想：不要给每个频率独立死记参数，而是用共享网络根据频率和条件生成 kernel/operator。
- 高频增强不是单纯靠 U-Net，而是希望物理条件、边界条件、频率条件能共同约束和恢复高频结构。

当前实验主线建议为：

```text
Stage 0: 数据与基线跑通
Stage 1: AM 思想增强 KNO 高频信息
Stage 2: KNO 小结构消融筛选
Stage 3: 参数化 K(u_n, c) 融入 KNO
Stage 4: PKNO + AM 融合
Stage 5: 高维复杂长时间 rollout 展示实验
```

其中 Stage 1 和 Stage 3 是真正主线；Stage 2 是辅助筛选；Stage 5 靠后，不作为早期开发门槛。

## 2. 数据可行性结论

### 2.1 优先级结论

优先选择数据时，按以下标准排序：

1. 数据已经公开，并且有现成 loader 或容易写 loader。
2. 能支持 KNO/AM-KNO 的高频指标与 rollout。
3. 能支持 PKNO 的参数化条件，例如 viscosity、forcing、Mach、边界条件、grid spacing、时间步长等。
4. 不要求一开始就处理超大真实数据。

推荐优先级：

| 优先级 | 数据 | 适合阶段 | 可行性判断 |
|---|---|---|---|
| A | FNO Burgers | Stage 0/1/2 | 最适合快速验证 mesh-independence、频谱误差、结构消融 |
| A | FNO Navier-Stokes | Stage 0/1 | 最适合 KNO baseline、长期 rollout、频谱误差 |
| A | PDEBench 1D/2D CFD 或 shallow-water | Stage 1/3 | 最适合 AM-FNO 对齐和后续参数化实验 |
| A- | 当前已有 AM-FNO 数据 | Stage 1 | 若本地服务器已有数据，应优先复用 |
| B | PKNN repo toy/generated systems | Stage 3 | 最适合验证 parameterized Koopman/shared dictionary 代码逻辑 |
| B | Rayleigh-Benard | Stage 4/5 | 适合长 rollout，但数据准备成本更高 |
| C | SEVIR water vapor | Stage 5 | 开源但大，适合最终真实数据展示 |
| C | Copernicus ocean current | Stage 5 | 开放但下载/预处理链条长，适合最后做真实系统 |

结论：早期不要先碰 SEVIR/Copernicus。先用 Burgers、Navier-Stokes、PDEBench/AM-FNO CFD 把模型、指标、训练代码跑稳。

### 2.2 数据源核验

| 数据源 | 公开情况 | 适合实验 | 备注 |
|---|---|---|---|
| FNO Burgers/Navier-Stokes | FNO 公开数据和生成脚本可用 | KNO baseline、mesh-independence、zero-shot resolution、long rollout | KNO 官方 README 也引用 NS demo 数据和 FNO 数据生成配置 |
| KoopmanLab demo data | 官方 README 给出 Google Drive 数据入口 | Navier-Stokes demo | 适合确认 KNO 原始训练流程 |
| PDEBench | 官方 GitHub 与 DaRUS 数据源公开 | CFD、shallow-water、跨参数/跨分辨率 | 更适合 PKNO 后续参数化条件实验 |
| AM-FNO 数据 | 当前仓库已有路径和复现指标记录 | AM-KNO 高频增强、CFD-1D/2D 对齐 | 需要确认数据来源和许可；大型 HDF5 不应提交 |
| PKNN/Parametric Koopman repo | 开源 repo 可用；用户反馈原代码基于 TensorFlow，repo 也可作为算法参考 | parameterized Koopman/shared dictionary 单元实验 | 不建议直接混 TF/PyTorch 训练，建议重写 PyTorch 模块 |
| SEVIR | AWS Open Data 公开 | 真实水汽长期预测 | 数据大，预处理成本高，不适合第一阶段 |
| Copernicus Marine | Copernicus Marine Data Store 免费开放 | 洋流长期预测 | 通常需要账号/API/区域裁剪，适合最终展示 |
| Rayleigh-Benard | 来源论文/代码可追溯 | zero-shot 时间间隔、长 rollout | 不如 FNO/PDEBench 直接，建议中后期 |

主要参考链接：

- KoopmanLab: https://github.com/Koopman-Laboratory/KoopmanLab
- FNO: https://github.com/scaomath/fourier_neural_operator
- PDEBench: https://github.com/pdebench/PDEBench
- PDEBench DaRUS dataset: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
- SEVIR AWS Open Data: https://registry.opendata.aws/sevir/
- Copernicus Marine Data Store: https://data.marine.copernicus.eu/
- Copernicus SEALEVEL product example: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description
- Parametric Koopman code reference: https://github.com/GUOYUE-Cynthia/Learning-Parametric-Koopman-Decompositions
- Turbulent Flow Nets / Rayleigh-Benard reference: https://github.com/Rui1521/Turbulent-Flow-Nets

## 3. TensorFlow PKNN 与 PyTorch KNO/AM-FNO 的兼容策略

当前困难：PKNN 相关开源实现主要不是围绕 PyTorch KNO/AM-FNO 生态写的，而 KNO 和 AM-FNO 都是 PyTorch。不要把 TensorFlow 模型直接塞进 PyTorch 训练循环。

推荐策略：

1. 把 PKNN repo 放进 `external/` 或 `ref/code/`，作为算法参考，不作为训练依赖。
2. 先重写最小 PyTorch 版本：
   - shared dictionary encoder `Psi_theta`
   - parameter-conditioned Koopman generator `K_phi(c)`
   - rollout/composition evaluator
   - reconstruction loss 与 prediction loss
3. 先在低维 toy system 上做 PyTorch 单元测试，确认公式和 TF 参考结果一致。
4. 再接入 KNO 频域分支，使 `K_k = G_phi(k, c, u_embed)`。
5. 避免跨框架共享 checkpoint；最多只迁移公式、初始化、数据生成方式和实验设置。

建议新项目里保留一个迁移说明：

```text
docs/pytorch_porting_notes_pknn.md
```

内容包括：

- TensorFlow 参考实现目录
- PyTorch 对应模块路径
- 每个模块的输入输出 shape
- 单元测试对照表
- 暂不迁移的部分

## 4. 实验主线与阶段设计

### Stage 0: 基线与数据管线

目标：不要先做复杂模型，先把数据、loader、metric、rollout、频谱分析全部跑通。

推荐数据：

- FNO Burgers
- FNO Navier-Stokes
- 当前 AM-FNO NS/CFD 数据

模型：

- KNO baseline
- FNO 或 AM-FNO baseline，如果已有复现代码

必须输出：

- `metrics.csv`
- `spectral_metrics.csv`
- `rollout_error_by_step.csv`
- `config.yaml`
- `env.txt`
- 若可行，保存少量预测图和频谱图

### Stage 1: AM-KNO 高频增强

目标：验证 AM 思想是否能改善 KNO 高频信息。

核心对照：

| 组别 | 模型 | 目的 |
|---|---|---|
| A0 | KNO original/fixed K | 纯 KNO baseline |
| A1 | KNO + frequency-conditioned K_k = f(k) | 看共享频率生成是否优于独立 K |
| A2 | KNO + AM-style K_k = f(k, u_embed) | 看当前状态条件是否改善高频 |
| A3 | KNO + AM high-frequency residual | 看高频 residual 是否有效 |
| A4 | A2 + A3 | 看 operator generation 与 residual 是否互补 |

推荐数据：

- Burgers：快速看频谱和 mesh-independence。
- NS-2D：看 rollout 与中高频涡结构。
- CFD-1D/2D：对齐 AM-FNO 数据，尤其检查变量尺度和高频误差。

关键指标：

```text
step relative L2
full rollout relative L2
low/mid/high band spectral relative error
gradient relative L2
rollout error growth slope
energy spectrum decay
```

### Stage 2: 小结构消融

目标：筛选结构，不要让它们抢主线。

建议只在 Stage 1 最优或次优配置上加小结构。

优先消融：

| 结构 | 版本 | 判断标准 |
|---|---|---|
| 线性残差 Koopman | `v_{t+1}=v_t + A v_t` | 若长 rollout 更稳，值得保留 |
| decoder-side FFN | `u = D(FFN(z))` 或 output refinement | 若高频改善且不破坏 rollout，可保留 |
| mini U-Net 高频分支 | output residual/refinement | 若同分辨率高频大幅提升但跨分辨率不掉太多，可保留 |
| post-Koopman FFN | `v_{t+1}=FFN(Kv_t)` | 谨慎；若短期提升但长期不稳，应排除 |

### Stage 3: Parameterized-KNO

目标：验证 `u_n / physical condition -> K` 是否能改善高频预测与时间复合。

推荐从小到大：

| 组别 | 模型 | 条件 |
|---|---|---|
| P0 | fixed KNO | 无条件 |
| P1 | `K_k=f(k)` | 频率条件 |
| P2 | `K_k=f(k, u_embed)` | 当前状态条件 |
| P3 | `K_k=f(k, c)` | 物理参数条件 |
| P4 | `K_k=f(k, u_embed, c, bc)` | 完整参数化 |

推荐数据：

- PKNN toy/generated systems：先验证 shared dictionary 与 parameterized Koopman family。
- PDEBench 参数化 PDE：如不同初始条件、参数、边界条件或 forcing。
- FNO NS 不同 viscosity：训练部分 viscosity，测试未见 viscosity。

关键实验：

```text
参数插值：train on ν = {1e-3, 1e-4}, test on ν = 5e-4
参数外推：train on mild cases, test on harder cases
长 rollout：train T_out=10, test T_out=40
跨分辨率：train 64x64, test 128x128
```

### Stage 4: PKNO + AM 融合

目标：把 AM 思想和 PKNO 合起来。

建议最终形式：

```text
z_t = Psi_theta(u_t, x, c)
u_embed = E_theta(u_t)
K_k = G_phi(freq_encoding(k), u_embed, c, bc)
z_{t+1,k} = K_k z_{t,k}
u_pred = D_theta(z_{t+1})
u_final = u_pred + optional_high_frequency_residual
```

核心验证：

- 比 fixed KNO 高频更好。
- 比 AM-KNO 跨物理条件更好。
- 比 Param-KNO 跨频率/跨分辨率更好。
- 长 rollout 误差增长更慢。

### Stage 5: 高维复杂长时间 rollout

放到靠后阶段。推荐数据：

- Rayleigh-Benard：适合长 rollout 和 zero-shot 时间间隔。
- SEVIR：真实水汽图像，适合最终展示。
- Copernicus ocean current：真实洋流，适合最终展示。

这些数据的意义更像 final demonstration，不适合第一阶段卡住开发。

## 5. 指标设计

基础指标：

```text
MSE
step relative L2
full rollout relative L2
rollout error by step
```

高频指标：

```text
FFT energy spectrum
low/mid/high frequency band relative error
high-frequency energy ratio error
spectral slope error
gradient relative L2
Laplacian relative L2
```

空间/边界指标：

```text
boundary-region relative L2
interior-region relative L2
edge/gradient-region error
```

动力学指标：

```text
rollout stability
error growth slope
energy drift
mass/conservation drift, if applicable
vorticity/enstrophy error, for fluid data
```

泛化指标：

```text
train resolution -> test resolution
seen physical parameter -> unseen physical parameter
short trained rollout -> long test rollout
seen boundary condition -> unseen/perturbed boundary condition
```

## 6. 推荐 GitHub 仓库结构

建议新项目直接按以下结构建库：

```text
pkno-experiments/
  README.md
  pyproject.toml
  .gitignore
  LICENSE

  configs/
    data_paths.example.env
    model/
      kno_base.yaml
      am_kno.yaml
      param_kno.yaml
      pkno_am.yaml
    experiment/
      burgers_mesh.yaml
      ns2d_rollout.yaml
      cfd1d_amkno.yaml
      pdebench_param.yaml

  docs/
    project_brief.md
    data_inventory.md
    experiment_protocol.md
    model_design_decisions.md
    pytorch_porting_notes_pknn.md
    pkno_experiment_preparation_report.md

  ref/
    papers/
    code_notes/
    data_cards/
    figures/
    equations/

  external/
    KoopmanLab/
    am_fno_repro/
    pknn_reference/
    PDEBench/

  data/
    raw/
    processed/
    index/

  src/
    pkno/
      __init__.py
      data/
        burgers.py
        navier_stokes.py
        pdebench.py
        amfno_cfd.py
      models/
        kno.py
        am_kno.py
        param_kno.py
        pkno.py
      operators/
        koopman_fixed.py
        koopman_frequency_conditioned.py
        koopman_parameterized.py
        complex_layers.py
      dictionaries/
        shared_dictionary.py
        encoders.py
        decoders.py
      highfreq/
        spectral_metrics.py
        conv_residual.py
        unet_residual.py
      losses/
        relative_lp.py
        spectral_loss.py
        conservation.py
      metrics/
        rollout.py
        spectral.py
        boundary.py
      trainers/
        train_rollout.py
        evaluate.py
      utils/
        seed.py
        logging.py
        checkpoint.py

  experiments/
    train.py
    eval.py
    ablate.py
    summarize_results.py

  scripts/
    download/
    preprocess/
    launch/
    collect_results.py

  tests/
    test_shapes.py
    test_complex_operator.py
    test_rollout.py
    test_spectral_metrics.py
    test_pknn_port.py

  outputs/
  results/
  reports/
```

`.gitignore` 必须忽略：

```text
data/
outputs/
checkpoints/
*.pt
*.pth
*.ckpt
*.hdf5
*.h5
*.mat
external/*/.git
```

## 7. `ref/` 目录建议放什么

为了让 Codex 后续更好地帮助模型设计、文件管理和实验追踪，`ref/` 不要只乱放 PDF。建议放成结构化参考库：

```text
ref/
  papers/
    KNO.pdf
    PKNN_or_parametric_koopman.pdf
    AM-FNO.pdf
    FNO.pdf
    F-FNO.pdf
    U-FNO_or_IU-FNO.pdf
    PDEBench.pdf

  code_notes/
    koopmanlab_code_map.md
    amfno_code_map.md
    pknn_tf_to_torch_map.md
    neuraloperator_api_notes.md

  data_cards/
    burgers.md
    navier_stokes.md
    pdebench_cfd.md
    shallow_water.md
    sevir.md
    copernicus.md

  equations/
    kno_forward.md
    parametric_koopman_forward.md
    am_kno_operator_generation.md
    pkno_forward.md

  figures/
    model_sketches/
    experiment_flow/
```

每个 `data_cards/*.md` 建议包含：

```text
数据来源 URL
许可/开放性
下载方式
原始 shape
处理后 shape
物理变量
物理参数
边界条件
适合的实验阶段
已知坑
```

每个 `code_notes/*.md` 建议包含：

```text
源码仓库 URL
关键文件
关键类/函数
输入输出 shape
依赖框架
可复用部分
需要重写部分
```

## 8. 给新 Codex 项目的起步文件

建议新项目初始化后，第一批文件就放：

```text
README.md
docs/project_brief.md
docs/data_inventory.md
docs/experiment_protocol.md
docs/model_design_decisions.md
docs/pytorch_porting_notes_pknn.md
configs/data_paths.example.env
configs/model/kno_base.yaml
configs/model/am_kno.yaml
configs/model/param_kno.yaml
configs/model/pkno_am.yaml
tests/test_shapes.py
tests/test_complex_operator.py
tests/test_rollout.py
```

给 Codex 的项目 brief 建议写清楚：

```text
本项目目标是做 PKNO/AM-KNO 实验，不是完整复现 KNO 论文。
优先数据：Burgers, Navier-Stokes, PDEBench/AM-FNO CFD。
第一阶段目标：跑通 KNO baseline + 高频频谱指标。
第二阶段目标：AM-style frequency-conditioned K。
第三阶段目标：Parameterized K(u_n, c)。
TensorFlow PKNN 只作为参考，不作为训练依赖。
所有新模型必须 PyTorch 实现。
所有数据/checkpoint/output 不提交 Git。
```

## 9. 近期行动清单

第一周建议：

1. 新建仓库结构。
2. 放入 `ref/` 文献和代码说明。
3. 写 `data_inventory.md`。
4. 跑通 Burgers loader + KNO baseline。
5. 实现 spectral metrics。
6. 对 NS-2D 做 1 epoch smoke test。
7. 把当前 AM-FNO CFD 数据路径迁移到 `configs/data_paths.env`。

第二周建议：

1. 实现 AM-style frequency-conditioned Koopman layer。
2. 跑 Burgers mesh/zero-shot resolution。
3. 跑 NS-2D rollout 10/40 steps。
4. 跑 CFD-1D/2D 高频谱误差。
5. 做固定 K vs `K_k=f(k)` vs `K_k=f(k,u_embed)` 消融。

第三周建议：

1. 实现 PyTorch shared dictionary + parameterized K prototype。
2. 用 PKNN toy/generated system 做单元实验。
3. 接入 PDEBench 参数化数据。
4. 测试跨参数泛化。

## 10. 当前最稳的实验路线

最稳路线如下：

```text
1. Burgers:
   KNO baseline -> AM-KNO -> spectral/mesh test

2. Navier-Stokes:
   KNO baseline -> AM-KNO -> long rollout/high-frequency spectrum

3. AM-FNO CFD or PDEBench CFD:
   AM-KNO 与 AM-FNO 口径对齐 -> 高频增强验证

4. PKNN toy/PDEBench param:
   shared dictionary + parameterized K 的 PyTorch 原型

5. PKNO + AM:
   条件化 K_k 生成 + shared dictionary + 高频 residual
```

这条路线的好处是：每一步都有可找到的数据，每一步都能独立产出结果，并且不会因为 PKNO 复杂代码还没完全实现而停住。

## 11. 最终结果表格与复杂度图模板

本节用于新项目后续直接填入实验结果。建议在 `reports/` 中保留最终表格，在 `results/` 中保留对应 CSV 源文件。所有表格都应该能从 CSV 自动生成，避免手工复制错误。

### 11.1 主实验对比表

主实验表建议用于回答：AM-KNO/Param-KNO/PKNO 是否相对 KNO、FNO、AM-FNO 等基线真正提升了高频信息、长期 rollout 和跨分辨率/跨条件泛化。

| Model | Core mechanism | Data | Train -> Test setting | Rollout steps | Step Rel L2 ↓ | Full Rel L2 ↓ | High-band spectral error ↓ | Gradient Rel L2 ↓ | Energy/physics drift ↓ | Params (M) | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| FNO | Fixed Fourier kernel | TBD | same-res / cross-res / cross-param | TBD | TBD | TBD | TBD | TBD | TBD | TBD | baseline |
| AM-FNO | Amortized Fourier kernel | TBD | same-res / cross-res / cross-param | TBD | TBD | TBD | TBD | TBD | TBD | TBD | AM reference |
| KNO | Fixed frequency Koopman matrix | TBD | same-res / cross-res / cross-param | TBD | TBD | TBD | TBD | TBD | TBD | TBD | KNO baseline |
| AM-KNO | AM-style frequency-conditioned Koopman | TBD | same-res / cross-res / cross-param | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Stage 1 main |
| Param-KNO | `K_k=f(k,u_embed,c)` | TBD | seen -> unseen condition | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Stage 3 main |
| PKNO + AM | shared dictionary + parameterized AM Koopman | TBD | seen -> unseen condition | TBD | TBD | TBD | TBD | TBD | TBD | TBD | final candidate |

建议把主实验拆成三张结果表或一张表的三个 block：

- **同分辨率长期 rollout**：看模型基本预测能力。
- **跨分辨率/mesh-independence**：看是否保持 neural operator 属性。
- **跨物理条件/参数泛化**：看 PKNO 的参数化 Koopman family 是否有效。

最重要的列不是单步误差，而是 `Full Rel L2`、`High-band spectral error` 和 `Energy/physics drift`。如果 AM-KNO 只降低单步 L2，但高频或长 rollout 不稳定，不能算主线成功。

### 11.2 消融实验表

消融表建议用于回答：哪些模块真的值得进入最终模型。小结构优化不要和主贡献混在一起，应该作为清晰的 keep/drop 决策表。

| Ablation ID | Base model | Changed module | Variant | Hypothesis | Step Rel L2 ↓ | Full Rel L2 ↓ | High-band spectral error ↓ | Rollout slope ↓ | Peak GPU memory | Keep? | Decision reason |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| B0 | KNO | none | fixed `K_k` | baseline | TBD | TBD | TBD | TBD | TBD | yes | reference |
| B1 | KNO | Koopman update | `v_{t+1}=v_t+A v_t` | residual linear update improves stability | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B2 | AM-KNO | frequency generator | `K_k=f(k)` | shared frequency generator improves high frequency | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B3 | AM-KNO | condition generator | `K_k=f(k,u_embed)` | current-state conditioning improves rollout | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B4 | Param-KNO | physical condition | `K_k=f(k,c)` | physical parameters improve cross-condition | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B5 | PKNO | shared dictionary | off/on | shared observables improve parameterized Koopman | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B6 | AM-KNO | high-frequency branch | conv vs mini U-Net | U-Net improves local high frequency | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B7 | AM-KNO | post-Koopman adapter | none vs FFN | nonlinear adapter improves short-term expression | TBD | TBD | TBD | TBD | TBD | TBD | drop if rollout unstable |

推荐判定规则：

```text
Keep:
  Full Rel L2 improves
  High-band spectral error improves
  Rollout slope does not get worse
  Cross-resolution or cross-condition performance does not collapse

Drop:
  Only one-step error improves
  Long rollout becomes unstable
  High-frequency energy becomes nonphysical
  Complexity increase is much larger than accuracy gain
```

### 11.3 计算复杂度与超参数表

复杂度表建议和主实验表分开。主实验表讲效果，复杂度表讲代价。这样能判断某个模块是不是“贵但不值”。

| Model | Operator type | `o` | modes | `r/decompose` | Generator hidden dim | HF branch | Batch size | Params (M) | Train sec/epoch | GPU-hours | Peak memory (GB) | Inference ms/step | Rollout ms/40 steps | Best checkpoint epoch |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KNO | fixed complex `K_k` | TBD | TBD | TBD | n/a | conv | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AM-KNO | `K_k=f(k)` | TBD | TBD | TBD | TBD | conv/residual | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Param-KNO | `K_k=f(k,u,c)` | TBD | TBD | TBD | TBD | conv/residual | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PKNO + AM | shared dictionary + generated `K_k` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| U-Net variant | refinement-heavy | TBD | TBD | TBD | TBD | mini U-Net | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

建议复杂度 CSV 字段：

```text
run_name, model, dataset, resolution, rollout_steps, o, modes, decompose,
generator_hidden_dim, generator_layers, hf_branch, batch_size,
params, train_seconds_per_epoch, total_gpu_hours, peak_memory_gb,
inference_ms_per_step, inference_ms_per_rollout,
step_rel_l2, full_rel_l2, high_band_spectral_error
```

### 11.4 复杂度-效果图设计

建议最终至少做一张复杂度-效果散点图。横轴是计算代价，纵轴是长期/高频误差，点大小是参数量，颜色表示模型家族。

![Complexity tradeoff template](assets/complexity_tradeoff_template.png)

图的设计建议：

| Element | Meaning | How to fill |
|---|---|---|
| x-axis | 计算代价 | 可用 GPU-hours、inference ms/step、rollout ms/40 steps 或 peak memory |
| y-axis | 效果指标 | 推荐用 `Full Rel L2` 或 `High-band spectral error`，越低越好 |
| bubble size | 参数量 | `Params (M)` |
| color | 模型家族 | KNO、AM-KNO、Param-KNO、PKNO+AM、U-Net variant |
| text label | 关键超参数 | 标注 `o/modes/r` 或 `hidden_dim/layers` |
| Pareto line | 候选最优模型 | 同时考虑低误差与可接受代价 |

可选做第二张图：超参数敏感性热力图。

```text
x-axis: modes
y-axis: operator size o
color: High-band spectral error or Full Rel L2
separate panels: r = 4 / 8 / 16
```

如果时间有限，优先做复杂度-效果散点图；热力图可以作为补充材料。
