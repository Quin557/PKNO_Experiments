# 数据目录说明

本目录用于在本地或服务器上放置实验数据。

当前第一阶段最重要的数据目录：

```text
data/
  navier_stokes/
    ns_V1e-3_N5000_T50.mat
    ns_V1e-4_N10000_T30.mat
  burgers/
    burgers_data_R10.mat
```

注意：

- `.mat`、`.h5`、`.hdf5`、`.npy` 等大数据文件不会提交到 git。
- 本目录里的 README 和 `.gitkeep` 只是为了保留目录结构。
- 服务器上也可以不把 `DATA_ROOT` 设置成仓库内的 `data/`，但目录结构建议保持一致。
