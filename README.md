# Molecular Symphony

**Harmonizing clinical and genetic data to enhance the precision and efficiency of glioma diagnosis.**

## Project Overview

Gliomas are the most common primary tumors of the brain, classified as either Lower-Grade Gliomas (LGG) or Glioblastoma Multiforme (GBM) based on histological and imaging criteria. Accurate grading is essential for effective treatment planning and prognosis. While clinical factors play a significant role in diagnosis, molecular and genetic features provide critical insights that can improve accuracy. However, molecular tests are often expensive, necessitating an optimal selection of features to balance cost and performance.

The **Molecular-Symphony** project aims to develop predictive models that determine whether a glioma is "LGG" or "GBM" using a combination of clinical and genetic data. By identifying the most informative subset of mutation genes and clinical features, this project seeks to enhance diagnostic precision while minimizing costs.

## Key Objectives

1. **Environment Setup**: Configure the development environment using IntelliJ IDEA, Scala, sbt, Java, and Hadoop.
2. **Data Loading**: Efficiently load the dataset into Scala for further processing.
3. **Data Preprocessing**: Preprocess the dataset to handle null values, encode categorical data, and ensure data quality.
4. **Feature Selection**: Utilize univariate feature selection techniques to identify the most relevant mutation genes and clinical features.
5. **Model Development**: Build a variety of machine learning and ensemble learning models to predict glioma grades based on the selected features.
6. **Model Evaluation**: Assess the performance of the developed models using metrics such as accuracy, precision, recall, and F1 score.


## Dataset Features

The dataset used in this project is "Glioma Grading Clinical and Mutation Features," comprising data from the TCGA-LGG and TCGA-GBM brain glioma projects. It includes the most frequently mutated 20 genes and 3 critical clinical features relevant to glioma grading. [Link to Dataset](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)

| Variable Name   | Role      | Type       | Description                                               | Values                        |
|-----------------|-----------|------------|-----------------------------------------------------------|-------------------------------|
| Grade           | Target    | Categorical| Glioma Grade Class Information                           | LGG, GBM                      |
| Gender          | Feature   | Categorical| Gender                                                    | male, female                  |
| Age_at_diagnosis| Feature   | Continuous | Age at diagnosis with the calculated number of days       | Numeric                       |
| Race            | Feature   | Categorical| Race                                                      | white, black or African American, Asian, American Indian or Alaska Native |
| IDH1            | Feature   | Categorical| Isocitrate Dehydrogenase (NADP(+)) 1                     | NOT_MUTATED, MUTATED          |
| TP53            | Feature   | Categorical| Tumor Protein p53                                         | NOT_MUTATED, MUTATED          |
| ATRX            | Feature   | Categorical| ATRX Chromatin Remodeler                                 | NOT_MUTATED, MUTATED          |
| PTEN            | Feature   | Categorical| Phosphatase and Tensin Homolog                            | NOT_MUTATED, MUTATED          |
| EGFR            | Feature   | Categorical| Epidermal Growth Factor Receptor                          | NOT_MUTATED, MUTATED          |
| CIC             | Feature   | Categorical| Capicua Transcriptional Repressor                         | NOT_MUTATED, MUTATED          |
| MUC16           | Feature   | Categorical| Mucin 16, Cell Surface Associated                         | NOT_MUTATED, MUTATED          |
| PIK3CA          | Feature   | Categorical| Phosphatidylinositol-4, 5-Bisphosphate 3-Kinase Catalytic Subunit Alpha | NOT_MUTATED, MUTATED |
| NF1             | Feature   | Categorical| Neurofibromin 1                                           | NOT_MUTATED, MUTATED          |
| PIK3R1          | Feature   | Categorical| Phosphoinositide-3-Kinase Regulatory Subunit 1           | NOT_MUTATED, MUTATED          |
| FUBP1           | Feature   | Categorical| Far Upstream Element Binding Protein 1                    | NOT_MUTATED, MUTATED          |
| RB1             | Feature   | Categorical| RB Transcriptional Corepressor 1                          | NOT_MUTATED, MUTATED          |
| NOTCH1          | Feature   | Categorical| Notch Receptor 1                                          | NOT_MUTATED, MUTATED          |
| BCOR            | Feature   | Categorical| BCL6 Corepressor                                          | NOT_MUTATED, MUTATED          |
| CSMD3           | Feature   | Categorical| CUB and Sushi Multiple Domains 3                          | NOT_MUTATED, MUTATED          |
| SMARCA4         | Feature   | Categorical| SWI/SNF Related, Matrix Associated, Actin Dependent Regulator of Chromatin, Subfamily A, Member 4 | NOT_MUTATED, MUTATED |
| GRIN2A          | Feature   | Categorical| Glutamate Ionotropic Receptor NMDA Type Subunit 2A       | NOT_MUTATED, MUTATED          |
| IDH2            | Feature   | Categorical| Isocitrate Dehydrogenase (NADP(+)) 2                     | NOT_MUTATED, MUTATED          |
| FAT4            | Feature   | Categorical| FAT Atypical Cadherin 4                                   | NOT_MUTATED, MUTATED          |
| PDGFRA          | Feature   | Categorical| Platelet-Derived Growth Factor Receptor Alpha            | NOT_MUTATED, MUTATED          |


## Project Setup

To set up the project environment, follow these steps:

1. **IDE**: The project is developed using IntelliJ IDEA Community Edition. Ensure you have IntelliJ IDEA installed on your system.

2. **SBT and Scala Versions**:
   - The project is built using SBT version 1.9.7 and Scala version 2.13.12.
   - Make sure you have SBT (Simple Build Tool) version 1.9.7 and Scala version 2.13.12 installed on your machine.
   - Additionally, ensure you have JDK 8 installed.

3. **Spark and Hadoop Versions**:
   - For data processing and model building, Spark version 3.5.0 is required.
   - Hadoop version 3.3.5 is needed for data loading and storage operations like svaing and trained models and processed data.

4. **Build.sbt Configuration**:
   - Below is the content of the `build.sbt` file:

    ```scala
    ThisBuild / version := "0.1.0-SNAPSHOT"

    ThisBuild / scalaVersion := "2.13.12"

    lazy val root = (project in file("."))
    .settings(
        // Project name and IDE package prefix
        name := "MolecularSymphony"
    )

    // Define versions for Spark and Hadoop
    val spark_version = "3.5.0"
    val hadoop_version = "3.3.5"

    // Define library dependencies
    libraryDependencies ++= Seq(
    // Spark dependencies
    "org.apache.spark" %% "spark-core" % spark_version,
    "org.apache.spark" %% "spark-sql" % spark_version,
    "org.apache.spark" %% "spark-mllib" % spark_version,

    // Hadoop dependencies
    "org.apache.hadoop" % "hadoop-common" % hadoop_version,
    "org.apache.hadoop" % "hadoop-client" % hadoop_version,
    "org.apache.hadoop" % "hadoop-hdfs" % hadoop_version
    )
    ```

5. **Dependency Management**:
    - Ensure that the specified versions of Spark and Hadoop dependencies are compatible with your project requirements.
    - SBT will automatically manage and download the specified dependencies when you build the project

## Data Preprocessing Guide

