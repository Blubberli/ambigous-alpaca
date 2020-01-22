### Readme test data - multiclass classification
This directory contains data that can be used to test aspects multiclass classification. Three different labels are annotated: **Form**, **Farbe** and **Dimension**.
There are three files containing training, test and validation data:
- test.txt (30 instances)
- train.txt (105 instances)
- val.txt (15 instances)
Each dataset contains three columns:

context [TAB] phrase [TAB] label

The phrases are unique for each split (a phrase that is in train is not in test or val)
