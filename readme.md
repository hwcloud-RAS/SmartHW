# Memory Failure Prediction Competition

### Background
<p style="font-size: 16px; line-height: 1.6;">
Uncorrectable Errors (UEs) of Dynamic Random Access Memory (DRAM) have been identified as a major failure cause in data centers. The multi-bit UE failure of High Bandwidth Memory (HBM) highly threatens the availability and reliability of servers and entire computing clusters. Forecasting UEs before enacting preemptive maintenance measures has emerged as a viable strategy for diminishing server outages. Some machine-learning based solutions have also been proposed.

However, the UEs prediction presents several challenges: data noise and extreme imbalance as the UEs are exceedingly rare in memory events; heterogeneous data sources as the DRAMs in the field come from different manufacturing or architecture platforms; distribution shifts due to hardware aging; and latent factors due to the dynamic access mechanism.

We cure a real-world memory error dataset that contains both micro and bit information and present a two-stage challenge for more efficient and generalized event prediction solutions. We believe the competition will provide a breeding ground to foster discussions and further progress on several important research topics towards real-world ML applications.
</p>

### Goal
<p style="font-size: 16px; line-height: 1.6;">
Our competition will furnish participants with a dataset comprising memory system configurations, memory error logs, and failure tags. This dataset will enable participants to devise solutions for predicting potential failures of individual DRAM modules within a subsequent observation period.

The competition comprises two stages. The initial stage features an AB List setup, which includes training data tailored for two diverse memory models. Subsequently, in the second stage, a fresh dataset encompassing mixed models (more than two) will be introduced. This encourages solutions with few-shot learning capabilities and knowledge transfer ability.

Overall, the competition’s appeal lies in its practical relevance, the accessible entry point of the initial stage, and the fresh challenges presented in both stages.
</p>

### Memory Architecture, Access, and Mitigation
<p style="font-size: 16px; line-height: 1.6;">
**DRAM components and errors**: Figure 1 illustrates the DRAM organization within a server, where the basic unit of installation is a dual in-line memory module (DIMM). At a fundamental level, a DIMM consists of multiple DRAM chips grouped into ranks, enabling simultaneous access during DRAM read/write operations within the same rank. Each chip contains multiple banks that operate in parallel. These banks are further divided into rows and columns, with the intersection of a row and column constituting a cell capable of storing a single data bit. Cells can store multiple bits, and the data width of a chip denoted as x4, x8, or x16, signifies the number of data bits stored in a cell. A DRAM error occurs when the DRAM exhibits abnormal behavior, resulting in one or more bits being read differently from their written values [5]. Modern DRAM implementations utilize error-correcting codes (ECC) to safeguard against DRAM errors.
</p>
<img src="https://raw.githubusercontent.com/qiaoyu0747/Memory_failure-prediction_competition/main/mem_structure.jpg" width="600" style="display: block; margin: auto;">
<p style="font-size: 16px; line-height: 1.6;" align="center">
  Figure 1: Memory Organization.
</p>
<p style="font-size: 16px; line-height: 1.6;">
**Memory access and RAS**: Figure 2(2) depicts the transmission process of x4 DRAM Double Data Rate 4 (DDR4) chips via DQs. Upon initiating a data request, 8 beats each with 72 bits (64 data bits and 8 ECC bits) including ECC error codes are transferred to memory controller via DQ wires. Implementing the contemporary ECC [6], [27], 72-bit data are spread across 18 DRAM chips, allowing the memory controller to detect and correct them with ECC in Figure 2(3). Note that ECC checking bits addresses are decoded to locate specific errors in DQs and beats. Then, all these logs including error detection and correction, events, and memory specifications are archived in Baseboard Management Controller (BMC) in Figure 2(4). Utilizing memory failure prediction in Figure 2(5) allows for the prediction of failures and the activation of corresponding mitigation techniques in Figure 2(6) based on specific use cases.
	</p>
	
<img src="https://raw.githubusercontent.com/qiaoyu0747/Memory_failure-prediction_competition/main/mem_organization.jpg" width="800" style="display: block; margin: auto;">
<p style="font-size: 16px; line-height: 1.6;" align="center">
  Figure 2: Memory Architecture, Access, and Mitigation Framework.
</p>
### Evaluation
<p style="font-size: 16px; line-height: 1.6;">
As we aim to predict whether each DRAM will occur UE failure or not within the next k days, the problem is formulated as a binary classification problem. The evaluation protocol is illustrated in Figure 3, derived from the production needed. Specifically, at present t, an algorithm observes historical data from an observation window △td to predict failures within the prediction period [t + △tl, t + △tl + △tp], where △tl is a minimum time interval between the prediction(i.e. lead time) and the failure. △tp denotes the prediction interval. We fix the lead prediction window to 15min (tl) and prediction windows is set as 7 days (△tp) for evaluation, but contestants are free to explore the observation window and labeling method for training.

A True Positive (TP) is a correctly predicted failure within the prediction window, while a False Positive (FP) is an incorrect prediction. A failure without a prior alarm is a False Negative (FN), and a True Negative (TN) occurs when no failures are predicted or occur. We assess the algorithm using F-score.
F − score = (1 + β2) Precision × Recall/β2 × Precision + Recall , (1)
where Precision = TP/TP+FP , Recall = TP/TP+FN , and β is a hyperparameter to balance the precision and recall. When β = 1, it becomes the F1-Score, at this point both recall and precision are important with equal weight. In cases where we consider precision more important, we set β < 1, vice verse.
</p>
<img src="https://raw.githubusercontent.com/qiaoyu0747/Memory_failure-prediction_competition/main/evaluation_framework.png" width="600" style="display: block; margin: auto;">
<p style="font-size: 16px; line-height: 1.6;" align="center">
	<p style="font-size: 16px; line-height: 1.6;" align="center">
  Figure 3: Evaluation Protocol.
</p>
