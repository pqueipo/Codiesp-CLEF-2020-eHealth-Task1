# Documentation of Master's final project 

In Healthcare domain, clinical case studies are provided in a narrative way with an unstructured format. They describe the patient’s conditions with natural language, making
the automated processing of such texts hard and challenging. In order to analyze and transform medical narratives into a structured or coded format, clinical coding is required.
Clinical coding is a crucial task for standardizing medical texts, monitor health trends and medical reimbursement. It is very critical for hospitals, insurance companies and
governments.

This work addresses the task of automatically assigning codes from the International Classification of Diseases, 10th version (ICD-10) for Diagnostics to unstructured Spanish
clinical case studies that were also translated into Spanish and evaluating the results.

This document presents an approach based on Named Entity Recognition (NER) to detect diagnoses and semantic linking relying on a terminological resource to extract the
labels. Each label is an ICD-10 code.

Precision results have been the best results obtained in the CodiEsp 2020 competition with a result of 86.6 %. The results of recall and Mean of Average Precision were 0.066
and 0.115 respectively.

The results are promising, especially precision and the evaluation regardless of the codes’ sub-category.

This work was supported by the Research Program of the Ministry of Economy and Competitiveness - Government of Spain, (DeepEMR project TIN2017- 87548-C2-1-R).

Keywords: ICD-10-CM * Clinical case studies * Multilabel classification * NamedEntity Recognition * Dictionary based approach * Fuzzy matching

