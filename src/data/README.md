# Dataset examples

Since both datasets are governed by strict data-use agreements prohibiting redistribution, we leave here only the skeleton of the data structure. Access requires a formal application to the respective data custodians.

## FOR2107

A German multicenter cohort study focused on the neurobiology of affective disorders. It comprises patients with Major Depressive Disorder (MDD) and matched healthy controls, with deep phenotyping spanning structural MRI, clinical assessments, neuropsychological testing, and demographic information.

**Classification task:** active MDD vs. healthy controls (binary).

Prior work on this cohort found classification accuracies of only 54–56% with univariate neuroimaging markers, and no informative individual-level biomarker even under extensive multivariate optimization across 4 million models - establishing FOR2107 as a genuinely hard classification problem.

## OASIS-3

An open-access longitudinal dataset compiled from the Washington University Knight Alzheimer Disease Research Center. It includes participants ranging from cognitively normal adults to individuals at various stages of cognitive decline, accompanied by multimodal MR sessions and clinical assessments.

**Classification task:** cognitive decline vs. cognitively normal (binary).

## Data preparation

Clinical variables are drawn from multiple CSV files, retaining the most recent value per participant. In consultation with domain experts in clinical psychology, trivially discriminative features (e.g. suicidal ideation) were excluded to ensure models must reason rather than pattern-match on near-perfect predictors.