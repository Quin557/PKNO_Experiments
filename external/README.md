# 外部源码目录

本目录用于放置本地克隆的第三方源码，方便阅读、对齐官方 baseline 和做算法参考。

推荐源码：

```bash
git clone https://github.com/Koopman-Laboratory/KoopmanLab external/KoopmanLab
git clone https://github.com/GUOYUE-Cynthia/Learning-Parametric-Koopman-Decompositions external/pknn_reference
git clone https://github.com/pdebench/PDEBench external/PDEBench
```

当前优先级：

1. `KoopmanLab`：KNO 官方 baseline 对齐。
2. `pknn_reference`：后续 parameterized Koopman / shared dictionary 参考。
3. `PDEBench`：后续参数化 PDE benchmark。

不要提交第三方源码仓库本体。
