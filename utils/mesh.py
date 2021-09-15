"""
How to build MESH DAG network. see this paper: 
Zhen-Hao Guo, Zhu-Hong You, De-Shuang Huang, Hai-Cheng Yi, Kai Zheng, Zhan-Heng Chen, Yan-Bin Wang, 
MeSHHeading2vec: a new method for representing MeSH headings as vectors based on graph embedding algorithm, 
Briefings in Bioinformatics, , bbaa037, https://doi.org/10.1093/bib/bbaa037




The MeSH consists of three parts including Main Headings, Qualifiers and Supplementary Concepts. Main Headings as the trunk of MeSH are used to describe the content or theme of the article. Qualifiers is the refinement of MeSH headings, i.e. how to be processed when it is in a specific area. Supplementary Concept is a complementary addition that is mostly related to drugs and chemistry. 

In MeSH tree structure, MeSH headings are organized as a ‘tree’ with 16 top categories in which the higher hierarchy has the broader meaning and the lower hierarchy has the specific meaning

Hence, we construct the MeSH heading relationship network from tree structure through hierarchical tree num rules.



Each MeSH heading can be described by one or more tree nums to reflect its hierarchy in the tree structure and relationships with other MeSH headings. Tree num consists of letters and numbers, the first of which is uppercase letter representing category and the rest are made up of numbers. The first two digits are fixed design following the first capital letter and can be seen the top category except capital letter.


 Each three digits represent a hierarchy in the tree structure. There are some MeSH headings such as Lung Neoplasms (C04.588.894.797.520, C08.381.540, and C08.785.520) that are described by a single type of tree num, while others such as Reflex (E01.370.376.550.650, E01.370.600.550.650, F02.830.702 and G11.561.731) can be represented by different kinds of tree num.

Whenever the last hierarchy of tree num is removed, a new tree num and corresponding MeSH heading can be generated and contacted.


 For the sake of simplicity, we treat the mode of the tree num category of MeSH heading as its label.

"""

import numpy as np
import pandas as pd
import networkx as nx



