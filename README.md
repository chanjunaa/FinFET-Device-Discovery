# FinFET-Device-Discovery

What is FinFET-Device-Discovery:   

*This machine learning-assisted tool is designed to discover complex 2D materials for FinFET devices*

---

## Documentation
As of the date of this release, the available documentation is contained in the FinFET-Device-Discovery/ directory.

## Installation
- Enviroments: Python
- Requirements:  
  -numpy  
  -pandas  
  -sklearn  
  -xgboost  
  -itertools  
  -matplotlib  
  
## Useage
- All original training data can be obtained from the "./data/FinFET_data.csv" file  
- classifymodel.py: Training a Classifier based on a suitable band gap range  
- confusion_matrix.py: For determining the performance of the Classifier  
- featureimportance.py: Get feature ranking results and plot  
- predict.py: Use the Classifier to predict band gaps for any number of other materials  
- All hypothetical tellurene strucutres data are available from the "data" folder  

## Having problems & Contact

chenan666@sjtu.edu.cn

## OurWebsite

www.aimslab.cn  
