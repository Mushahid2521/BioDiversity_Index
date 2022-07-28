## PyBioSci
**Py**thon **BioSci**ence is a package containing different functionalities on different BioScience algorithms. 
Initially the package will be focused on the Microbiome data analysis, later the functionalities will be expanded to other BioScience tasks. 


### Diversity Indices
This module contains different **alpha** and **beta** diversity indices to calculate the diversity of a sample and a group of samples. All the indices are discussed in this [paper](https://academic.oup.com/bib/article/19/4/679/2871295?login=false).

Below an example code for finding diversity indices has been shown: 

```
from pybiosci.diversity_indicies import alpha_index 
import pandas as pd

df = pd.read_csv('data/simCounts.csv')
print(alpha_index(df, index='Shannon'))
```

The dataframe is expected to be in a definite format. Row represents the taxa and column represents each sample. Absolute read counts are taken as input and converted to relative abundance when required for an index. You can specify `index=All` to get all indices as a feature table. 

