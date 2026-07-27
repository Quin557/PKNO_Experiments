# Burgers 数据目录

第三优先级下载：

```text
data/burgers/burgers_data_R10.mat
```

用途：

- 快速 KNO baseline；
- mesh 测试；
- 频谱指标验证。

对应服务器路径变量：

```bash
DATA_ROOT=/path/to/pkno_data
BURGERS_FILE=burgers/burgers_data_R10.mat
```

检查：

```bash
source configs/data_paths.env
ls -lh "$DATA_ROOT/$BURGERS_FILE"
```

数据文件不要提交到 git。
