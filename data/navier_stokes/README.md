# Navier-Stokes 数据目录

第一阶段 KNO baseline 优先使用这里的数据。

请把 KoopmanLab / KNO 官方数据文件放到本目录：

```text
data/navier_stokes/ns_V1e-3_N5000_T50.mat
data/navier_stokes/ns_V1e-4_N10000_T30.mat
```

对应服务器路径变量：

```bash
DATA_ROOT=/path/to/pkno_data
NS2D_V1E3_FILE=navier_stokes/ns_V1e-3_N5000_T50.mat
NS2D_V1E4_FILE=navier_stokes/ns_V1e-4_N10000_T30.mat
```

检查：

```bash
source configs/data_paths.env
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
```

数据文件不要提交到 git。
