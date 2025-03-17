Congratulations to all teams who successfully advanced to Phase 2 and completed the competition! To ensure smooth evaluation process, please submit complete materials as required for us to review the executability, reproducibility and compliance of your solutions. The specific submission requirements and final scoring rules are as follows:

# Code Submission Specifications

Teams passing Phase 2 qualification review must submit materials according to these specifications:

## Code File Requirements

1. **Code Project Directory** containing but not limited to:
    - `main.py`
      Main program implementing three processing pipelines: feature generation (`generate_features`), model training (`training_pipeline`), and inference (`inference_pipeline`). Template available at [main.py](https://github.com/hwcloud-RAS/SmartHW/blob/main/phase2_materials/main.py)
    - `requirements.txt` listing all third-party dependencies with versions
    - `README.md` technical documentation describing code architecture, module functions, and execution instructions

2. **Model & Data Archives** containing generated features, models and predictions:
    - `features.zip` from `arg.feature_path`
    - `model.zip` from `args.model_path` 
    - `submission.csv` prediction results file

# Technical Report Requirements

Technical report (English) should not exceed 4 pages for main content (unlimited appendices). Required sections:

1. **Problem Definition**  
   Formalize competition problem using mathematical modeling
2. **Technical Solution**  
   Explain solution using text, framework diagrams, and data flow charts
3. **Deployment Instructions**  
   Code execution flow, third-party dependencies, and optimal hyperparameters (no tuning during review)
4. **Solution Evaluation**  
   Objective analysis of innovation, effectiveness and limitations

Format must follow official LaTeX template: [Overleaf Link](https://www.overleaf.com/read/rbrtbpjcqrrg#a48d10)

# Submission Guidelines

1. **Code & Data Submission**  
   Deadline: Mar 18, 2025 20:00 (UTC+8)  
   Methods (choose one): Anonymous GitHub repo, Baidu Cloud, OneDrive  
   Submit via [Collection Form](XXX)

2. **Technical Report**  
   Deadline: Mar 23, 2025 20:00 (UTC+8)  
   Submit via [Openreview](https://openreview.net/group?id=ACM.org/TheWebConf/2025/Competition/SmartMem#tab-recent-activity)

# Evaluation Rules

## Verification Process

1. **Completeness Check**
    - Material completeness
    - Originality verification (plagiarism check)

2. **Code Compliance Review**
    - Code executability
    - Model specification compliance (time constraints, consistency)

3. **Functional Verification**
    - Consistency between code output and submitted data

Note: Failed completeness check leads to disqualification. Failed compliance review allows one code modification opportunity.

## Code Compliance Reminders

Common Phase 1 issues to avoid:
1. **Incorrect timestamp labeling** - Use latest timestamp from input samples
2. **Future information leakage** - Strictly prohibit using future data
3. **Immutable inference results** - No modification after inference
4. **Consistent SN processing** - Uniform processing framework for all SNs

# Final Scoring Formula

**Final Score = 0.8 × Reproduced Result Score + 0.2 × Paper Quality Score**

1. **Reproduced Result Score**  
   F1-score on Phase 2 test set (executed by organizers)

2. **Paper Quality Score**  
   Average score from ≥3 domain experts assessing innovation, technical soundness, and application value

We appreciate your outstanding contributions and wish you success in the final evaluation!

---

祝贺所有成功晋级第二阶段并顺利完赛的参赛团队！为确保评审工作顺利进行，请您按照提交完整材料，以便我们就方案的可执行性、可复现性与合规性进行审查。具体需要提交的材料与最终评分规则如下。

# 代码文件提交规范

通过第二阶段资格审查的参赛团队须按以下规范提交材料：

## 代码文件材料

1. **代码工程目录**：包含但不限于以下核心文件：
    - `main.py`
      主程序文件，完整实现特征生成（generate_features）、模型训练（training_pipeline）、推理（inference_pipeline）三类处理流程，模板详见 [main.py](https://github.com/hwcloud-RAS/SmartHW/blob/main/phase2_materials/main.py)
    - `requirements.txt`依赖清单，列明运行环境所需全部第三方依赖包及其版本
    - `README.md`技术说明文档，包含代码架构说明、模块功能描述及运行指引

2. **模型与数据压缩包**：包含由上述代码生成的特征数据、模型与预测结果：
    - `features.zip`特征数据压缩包，对应代码中 `arg.feature_path` 路径下的特征数据
    - `model.zip`模型文件压缩包，对应代码中 `args.model_path` 路径下的模型
    - `submission.csv`预测结果文件，即代码运行输出的故障预测结果文件

## 技术报告要求

技术报告（英文）正文篇幅不超过 4 页，附录篇幅不限。报告正文包含以下章节：

1. **问题定义**  
   结合竞赛背景阐述对问题的理解，建议通过数学建模方式进行形式化描述；
2. **技术方案**  
   建议综合运用文字说明、框架图示、数据流程图等方式系统阐述解决方案；
3. **部署说明**  
   明确代码运行流程、第三方依赖项、核心超参数设置等实施细节。请提供最优超参数配置，评审阶段不再进行参数调优；
4. **方案评价**  
   客观分析方案的创新性、有效性及局限性。

技术报告格式应该遵循组委会提供的 LaTeX 模板，详见 [Overleaf 链接](https://www.overleaf.com/read/rbrtbpjcqrrg#a48d10)。

## 材料提交说明

1. **代码与数据提交**  
   截止时间：2025年3月18日20时（北京时间，UTC+8）  
   提交方式（三选一）：匿名GitHub仓库上传、百度网盘存储、OneDrive存储
   请通过指定收集表（链接：XXX）提交存储链接

2. **技术报告**  
   截止时间：2025年3月23日20时（北京时间，UTC+8）  
   报告模板：[Overleaf 链接](https://www.overleaf.com/read/rbrtbpjcqrrg#a48d10)
   请通过 [Openreview 链接](https://openreview.net/group?id=ACM.org/TheWebConf/2025/Competition/SmartMem#tab-recent-activity) 提交技术报告

# 评审计分规则

## 材料核验流程

1. **完整性核验**
    - 材料完整性检查（代码文件、模型数据归档包、技术报告）
    - 参赛作品原创性验证（团队间材料重复率检测）

2. **代码合规性审查**
    - 代码是否能够成功运行
    - 代码是否符合模型构建规范（时间约束、一致性要求）

3. **功能性验证**
    - 代码输出与提交数据包一致性校验

请注意，如果您的材料未能通过完整性核验，将被取消参赛资格；如果您的代码未能通过代码合规性审查，我们将列举问题代码，您仅有一次代码修改机会，若修改后仍不能通过审查，将被取消参赛资格。

## 代码合规性审查注意事项

在一阶段的代码审查中，我们发现了一些常见的问题，这些问题可能会影响您在第二阶段的评审结果。以下为一些违反模型构建规范的代码案例，请您对照检查您的代码是否出现以下问题：

1. **生成样本时标注的时间戳有误**：生成样本时，其标注的时间戳应为所有输入样本中最新的时间戳；

2. **生成样本时预知未来的信息**：生成样本时，不能使用未来的信息；

3. **已经推理的结果不能再修改**：在推理过程中，不能再修改已经推理的结果；

4. **所有SN必须用一套处理框架**：在样本生成和模型训练时，不能人工指定不同 SN 采用不同方法处理；

## 最终评分标准

通过核验的作品将按以下公式计算最终得分：  
**最终得分 = 0.8 × 复现结果分数 + 0.2 × 论文质量分数**

1. **复现结果分数**  
   根据第二阶段测试集 F1-score 评定，该分数由组委会在测试集上运行通过核验的代码所得，排行榜上高分并不意味着最终得分高

2. **论文质量分数**  
   由三位以上领域专家从创新性、技术合理性、应用价值等维度进行评分后取均值

最后，感谢各位的卓越贡献与热忱参与，预祝在最终评审中斩获佳绩！