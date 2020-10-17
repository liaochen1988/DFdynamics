SCFA producer from MelonnPan analysis:

Samples:
All the stool samples with both 16S seq data and SCFA data from the inulin and RS groups (EXCEPT samples on day 0).
Performed at the lowest taxonomy level and asv level.


Method: 
MelonnPan - Model-based Genomically Informed High-dimensional Predictor of Microbial Community Metabolic Profiles

References:
Franzosa EA et al. (2019). Gut microbiome structure and metabolic activity in inflammatory bowel disease. Nature Microbiology 4(2):293–305.

Input:
MelonnPan-Train workflow requires the following inputs:
a table of metabolite relative abundances (samples in rows)
a table of microbial sequence features' relative abundances (samples in rows)

Output:
The separately analysis results of Inulin and RS group are stored in folder Inulin and RS, respectively. 
The combined analysis result of inulin and RS group are stored in folder Inulin+RS.

the MelonnPan-Train workflow outputs the following:
MelonnPan_Training_Summary.txt: Significant compounds list with per-compound prediction accuracy (correlation coefficient) and the associated p-value and q-value.
MelonnPan_Trained_Metabolites.txt: Predicted relative abundances of statisticially significant metabolites as determined by MelonnPan-Train.
MelonnPan_Trained_Weights.txt: Table summarizing coefficient estimates (weights) per compound.