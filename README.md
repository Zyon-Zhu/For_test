# Project_test

测试仓库 / 数据分析工作区。

## 本地环境

- Git 2.36.0.windows.1
- Python + pandas（分析脚本运行环境）
- 远程仓库：https://github.com/Zyon-Zhu/For_test.git
- 默认分支：main

## 目录结构

```
README.md            本说明
README_TEMPLATE.md   分析报告模板（填写规则见文件头）
AI_LOG.md            AI 交互日志（边用边记）
analyze.py           通用分析脚本骨架
data/                放原始数据（CSV）
outputs/             放脚本输出（summary.txt 等）
```

## 分析工作流

1. 把数据放到 `data/`
2. 运行 `python analyze.py data/xxx.csv --out outputs/summary.txt`
3. 复制 `README_TEMPLATE.md` 填写分析结论
4. **铁律**：README 里的每一个数字都必须能由脚本输出或明确步骤复算出来

## 常用命令

```bash
git pull origin main      # 拉取远程更新
git add -A                # 暂存所有变更
git commit -m "说明"      # 提交
git push origin main      # 推送
```
