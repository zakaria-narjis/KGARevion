<h1 align="center">
  KGARevion: Knowledge Graph Based Agent for Complex, Knowledge-Intensive QA in Medicine
</h1>

## 👀 Overview of KGARevion
Biomedical knowledge is uniquely complex and structured, requiring distinct reasoning strategies compared to other scientific disciplines like physics or chemistry. Biomedical scientists do not rely on a single approach to reasoning; instead, they use various strategies, including rule-based, prototype-based, and case-based reasoning. This diversity calls for flexible approaches that accommodate multiple reasoning strategies while leveraging in-domain knowledge. We introduce KGARevion, a knowledge graph (KG) based agent designed to address the complexity of knowledge-intensive medical queries. Upon receiving a query, KGARevion generates relevant triplets by using the knowledge base of the LLM. These triplets are then verified against a grounded KG to filter out erroneous information and ensure that only accurate, relevant data contribute to the final answer. Unlike RAG-based models, this multi-step process ensures robustness in reasoning while adapting to different models of medical reasoning. Evaluations on four gold-standard medical QA datasets show that KGARevion improves accuracy by over 5.2%, outperforming 15 models in handling complex medical questions. To test its capabilities, we curated three new medical QA datasets with varying levels of semantic complexity, where KGARevion achieved a 10.4% improvement in accuracy. 

![KGARevion framework](https://github.com/mims-harvard/KGARevion/blob/main/model_architecture.jpg)

## 🚀 Installation

1⃣️ First, clone the Github repository:

```bash
$ git clone https://github.com/mims-harvard/KGARevion
$ cd KGARevion
```

2⃣️ Then, set up the environment. This codebase leverages Python, Pytorch, Pytorch Geometric, etc. To create an environment with all of the required packages, please ensure that conda is installed and then execute the commands:

```bash
$ conda env create -f KGARevion.yaml
$ conda activate KGARevion
```

### 🛠️ Fine-tuning LLMs

After cloning the repository and installing all dependencies. You can run the following command to train our model:


### 🏙️ Figure 

To recover each figure presented in this paper, please download the code and data at [Fig.zip(google drive)](https://drive.google.com/file/d/1sCM8xh9tdyhAU0fHUPiyVbJwQeKvBwu2/view?usp=sharing) or [Fig.zip (zenodo)](https://zenodo.org/records/11554803). You could find all code and initial data in this folder to recover each figure. By the way, some figures are painted by GraphPad, so we also uploaded the initial file of GraphPad. 

You can also access these figures in this repo by clicking 'Fig' and run notebooks to get them.

### 🌟 Personalize based on your own dataset

If you want to benchmark KGARevion with your own QA dataset. You are kindly requested to prepare the following fil


### ⚖️ License

The code in this package is licensed under the MIT License.

</details>

