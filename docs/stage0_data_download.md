# Stage 0 数据下载与服务器放置说明

本文件只回答第一阶段的数据问题：

```text
第一阶段数据去哪找？
下载什么文件？
放到服务器哪里？
怎么检查路径是否正确？
```

当前第一阶段只按 **KoopmanLab / KNO 官方 baseline** 来准备数据，不额外考虑 FNO 数据生成仓库。

## 1. 数据来源优先级

第一阶段数据来源优先级：

1. KoopmanLab 官方 README 给出的数据入口；
2. KNO 论文中说明的数据；
3. 如果官方入口缺文件，再询问并手动补齐。

当前最重要的数据是：

```text
Navier-Stokes
```

原因：

- KoopmanLab 官方仓库自带 `demo_ns.py`；
- `demo_ns.py` 直接演示 KNO2d 在 Navier-Stokes 上的训练；
- 当前第一阶段 baseline 最容易先从这个官方 demo 跑通。

## 2. KoopmanLab 官方数据入口

KoopmanLab 官方 README 中给出的数据入口是 Google Drive 文件夹：

```text
https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
```

本地对应说明文件：

```text
ref/code_notes/koopmanlab_code_map.md
```

官方仓库位置：

```text
external/KoopmanLab
```

官方 demo：

```text
external/KoopmanLab/demo_ns.py
```

## 3. 第一批必须下载的文件

### 3.1 Navier-Stokes v1e-3

第一优先级下载：

```text
ns_V1e-3_N5000_T50.mat
```

原因：

- KoopmanLab 官方 `demo_ns.py` 默认使用这个文件名；
- 官方 demo 中路径写作：

```python
data_path = "./data/ns_V1e-3_N5000_T50.mat"
```

服务器目标位置：

```text
$DATA_ROOT/raw/navier_stokes/ns_V1e-3_N5000_T50.mat
```

### 3.2 Navier-Stokes v1e-4

第二优先级下载：

```text
ns_V1e-4_N10000_T30.mat
```

原因：

- KoopmanLab `kp.data.navier_stokes(..., type="1e-4")` 有对应分支；
- 适合作为另一组 KNO baseline。

服务器目标位置：

```text
$DATA_ROOT/raw/navier_stokes/ns_V1e-4_N10000_T30.mat
```

### 3.3 Burgers

第三优先级下载：

```text
burgers_data_R10.mat
```

说明：

- KoopmanLab 有 `kp.data.burgers(path, batch_size=64, sub=32)` 数据接口；
- 该数据适合快速 baseline、mesh 测试和频谱指标验证；
- 如果 KoopmanLab 数据文件夹里没有该文件，请先不要阻塞 Navier-Stokes baseline，后续再补。

服务器目标位置：

```text
$DATA_ROOT/raw/burgers/burgers_data_R10.mat
```

### 3.4 Shallow-water

第四优先级，暂缓：

```text
shallow_water_data.mat
```

说明：

- KoopmanLab 有 `kp.data.shallow_water(...)` 接口；
- 但具体官方文件名和来源需要进一步确认；
- 不作为第一轮 smoke test 的阻塞项。

服务器目标位置暂定：

```text
$DATA_ROOT/raw/shallow_water/shallow_water_data.mat
```

## 4. 服务器目录结构

推荐在服务器上建立：

```text
/path/to/pkno_data/
  navier_stokes/
    ns_V1e-3_N5000_T50.mat
    ns_V1e-4_N10000_T30.mat
  burgers/
    burgers_data_R10.mat
  shallow_water/
    shallow_water_data.mat
```

其中 Navier-Stokes 是当前第一阶段必须优先准备的。

仓库内也已经保留了对应目录模板：

```text
data/
  navier_stokes/
  burgers/
```

如果你希望直接把仓库内 `data/` 作为服务器数据根目录，可以设置：

```bash
DATA_ROOT=/absolute/path/to/PKNO_Experiments/data
```

## 5. 配置 `configs/data_paths.env`

在服务器仓库根目录执行：

```bash
cp configs/data_paths.example.env configs/data_paths.env
vim configs/data_paths.env
```

推荐内容：

```bash
DATA_ROOT=/path/to/pkno_data
KOOPMANLAB_ROOT=external/KoopmanLab

NS2D_V1E3_FILE=raw/navier_stokes/ns_V1e-3_N5000_T50.mat
NS2D_V1E4_FILE=raw/navier_stokes/ns_V1e-4_N10000_T30.mat

BURGERS_FILE=raw/burgers/burgers_data_R10.mat
SHALLOW_WATER_FILE=raw/shallow_water/shallow_water_data.mat
```

注意：

```text
configs/data_paths.env
```

是私有文件，不能提交到 git。

## 6. 检查数据是否放对

```bash
source configs/data_paths.env

ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
```

如果 Burgers 已下载：

```bash
ls -lh "$DATA_ROOT/$BURGERS_FILE"
```

如果 shallow-water 已下载：

```bash
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

## 7. 用 KoopmanLab 检查数据能否读取

先检查 Navier-Stokes v1e-3：

```bash
source configs/data_paths.env

python - <<PY
import sys
from pathlib import Path

sys.path.insert(0, str(Path("external/KoopmanLab").resolve()))
import koopmanlab as kp

path = Path("$DATA_ROOT/$NS2D_V1E3_FILE")
print("data:", path)
train_loader, test_loader = kp.data.navier_stokes(
    str(path),
    batch_size=10,
    T_in=10,
    T_out=40,
    type="1e-3",
    sub=1,
)
print("train batches:", len(train_loader))
print("test batches:", len(test_loader))
PY
```

## 8. 下载不到怎么办

如果 Google Drive 里找不到对应文件，或文件名不一致，请记录：

```text
文件夹里实际有哪些文件？
文件大小分别是多少？
是否包含 ns_V1e-3_N5000_T50.mat？
是否包含 ns_V1e-4_N10000_T30.mat？
```

然后把文件列表发回来，再决定：

- 是否改 `configs/data_paths.env`；
- 是否新增数据卡；
- 是否先只跑 v1e-3；
- 是否需要从 KNO 论文补找其它官方数据链接。

## 9. 当前最小可执行数据目标

只要先拿到下面这个文件，就可以启动第一阶段 smoke test：

```text
$DATA_ROOT/raw/navier_stokes/ns_V1e-3_N5000_T50.mat
```

然后按照：

```text
docs/server_run_checklist.md
```

运行：

```text
smoke_koopmanlab_ns_v1e3_gpu0
```
