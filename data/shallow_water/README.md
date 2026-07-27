# Shallow-water 数据目录

后续 KNO baseline / rollout 扩展实验使用这里的数据。

建议文件名：

```text
data/shallow_water/2D_rdb_NA_NA.h5
```

来源：

```text
PDEBench SWE / 2D radial dam break shallow-water dataset
```

对应服务器路径变量：

```bash
DATA_ROOT=/path/to/pkno_data
SHALLOW_WATER_FILE=shallow_water/2D_rdb_NA_NA.h5
```

检查：

```bash
source configs/data_paths.env
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

数据文件不要提交到 git。
