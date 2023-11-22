# Code Smell Detection Research Based on Pre-training and Stacking Models

Resources and extra documentation for the manuscript“Code Smell Detection Research Based on Pre-training and Stacking Models”published in IEEE Latin America Transactions.The project hierarchy and folders description is as follows.

1.bert.py is used to get the text information of the code.

2.birdelineSmote.py is used to add positive samples to the dataset

3.LDA.py is used to downscale our text information

4.REFCV.py is used to filter the dataset for metrics

5.stacking.py is the main part of the model

6.iplasma.zip is the tool used to extract the metric information

Code smells detection primarily adopts heuristicbased, machine learning, and deep learning approaches. However,to enhance accuracy, most studies employ deep learning methods,but the value of traditional machine learning methods should notbe underestimated. Additionally, existing code smells detection methods do not pay sufficient attention to the textual features in the code. To address this issue, this paper proposes a code smell detection method, SCSmell, which utilizes static analysis tools to extract structure features, then transforms the code into txt format using static analysis tools, and inputs it into the BERT pretraining model to extract textual features. The structure features are combined with the textual features to generate sample data and label code smells instances. The REFCV method is then used to filter important structure features. To deal with the issue of data imbalance, the Borderline-SMOTE method is used to generate positive sample data, and a three-layer Stacking model is ultimately employed to detect code smells. In our experiment, we select 44 large actual projects programs as the training and testing sets and conducted smell detection for four types of code smells: brain class, data class, God class, and brain method. The experimental results  indicate that the SCSmell method improves the average accuracy by 10.38% compared to existing detection methods, while maintaining high precision, recall, and F1 scores.The SCSmell method is an effective solution for  implementing code smells detection.The flowchart of the whole project is as follows.
![](https://markdown.liuchengtu.com/work/uploads/upload_4c5d630818dda7cede7cd5640bb97fa1.png)

