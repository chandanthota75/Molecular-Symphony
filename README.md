# Molecular Symphony

**Harmonizing clinical and genetic data to enhance the precision and efficiency of glioma diagnosis.**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Objectives](#key-objectives)
3. [Dataset Features](#dataset-features)
4. [Project Setup](#project-setup)
5. [Data Preprocessing Guide](#data-preprocessing-guide)
6. [Training the Models](#training-the-models)
7. [Evaluation Metrics](#evaluation-metrics)
8. [The Results](#the-results)
9. [Conclusion and Future Enhancement](#conclusion-and-future-enhancement)
10. [Code Overview](#code-overview)

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

Follow these steps to preprocess the data and select features for your model:

1. **Loading Necessary Columns**:
   - Use the `loadData` function to load the necessary columns from the dataset.

   ```scala
   def loadData(spark: SparkSession, filePath: String): DataFrame = {
     logger.info(s"Loading data from: $filePath")
     val data = loadCSV(spark, filePath)
     logger.info("Removing columns: Project, Case_ID")
     data.drop("Project", "Case_ID")
   }
   ```

2. **Define Schema**:
   - Defining the schmea for the dataframe to maintain data quality and ensure correct model training.
   ```scala
    private val schema: StructType = StructType(Array(
        StructField("Gender", IntegerType, nullable = false),
        StructField("Age_at_diagnosis", DoubleType, nullable = false),
        StructField("Primary_Diagnosis", IntegerType, nullable = false),
        StructField("Race", IntegerType, nullable = false),
        StructField("IDH1", IntegerType, nullable = false),
        StructField("TP53", IntegerType, nullable = false),
        StructField("ATRX", IntegerType, nullable = false),
        StructField("PTEN", IntegerType, nullable = false),
        StructField("EGFR", IntegerType, nullable = false),
        StructField("CIC", IntegerType, nullable = false),
        StructField("MUC16", IntegerType, nullable = false),
        StructField("PIK3CA", IntegerType, nullable = false),
        StructField("NF1", IntegerType, nullable = false),
        StructField("PIK3R1", IntegerType, nullable = false),
        StructField("FUBP1", IntegerType, nullable = false),
        StructField("RB1", IntegerType, nullable = false),
        StructField("NOTCH1", IntegerType, nullable = false),
        StructField("BCOR", IntegerType, nullable = false),
        StructField("CSMD3", IntegerType, nullable = false),
        StructField("SMARCA4", IntegerType, nullable = false),
        StructField("GRIN2A", IntegerType, nullable = false),
        StructField("IDH2", IntegerType, nullable = false),
        StructField("FAT4", IntegerType, nullable = false),
        StructField("PDGFRA", IntegerType, nullable = false),
        StructField("Label", IntegerType, nullable = false)
    ))
   ```
3. **Label Encoding**:
   - Perform label encoding for all categorical columns to prepare the data for model training.
   - Use mappings and UDFs to encode categorical values.

   ```scala
    val genderMapping = Map("Male" -> 0, "Female" -> 1)
    val raceMapping = Map(
      "white" -> 0,
      "black or african american" -> 1,
      "asian" -> 2,
      "american indian or alaska native" -> 3,
      "not reported" -> 4
    )
    val primaryDiagnosisMapping = Map(
      "Astrocytoma, anaplastic" -> 0,
      "Astrocytoma, NOS" -> 1,
      "Glioblastoma" -> 2,
      "Mixed glioma" -> 3,
      "Oligodendroglioma, anaplastic" -> 4,
      "Oligodendroglioma, NOS" -> 5
    )
    val mutationMapping = Map("NOT_MUTATED" -> 0, "MUTATED" -> 1)
    val gradeMapping = Map("LGG" -> 0, "GBM" -> 1)
   ```

4. **Feature Selection**:
   - Perform feature selection using the UnivariateFeatureSelector class to get features for training the model
   ```scala
    val selector = new UnivariateFeatureSelector()
    .setSelectionMode("numTopFeatures")
    .setSelectionThreshold(16)
    .setFeatureType("categorical")
    .setLabelType("categorical")
    .setFeaturesCol("features")
    .setLabelCol("Label")
    .setOutputCol("selectedFeatures")
   ```

5. **Vector Assembler and Train-Test Split**:
   - Finally, using vector and string assembler to assemble features into a single vector column.
   - Shuffle and Split the data into training and testing sets [80 : 20].

## Training the Models

I have used the below machine and ensemble learning models to predict the LGG and GBM from the dataset.

1. **Logistic Regression (LR)**:
   - Using the Logistic Regression algorithm to train the model. 
   - Configure parameters such as regularization and convergence tolerance as needed.

2. **Decision Tree (DT)**:
   - Training the Decision Tree model using the dataset.
   - Specify parameters such as maximum depth and minimum instances per node.

3. **Random Forest (RF)**:
   - Training the Random Forest model by building multiple decision trees.
   - Configure parameters such as the number of trees and maximum depth.

4. **Naive Bayes (NB)**:
   - Training the Naive Bayes classifier using the dataset.

5. **Support Vector Classifier (SVC)**:
   - Training the Support Vector Classifier using the dataset.
   - Configure parameters such as the kernel type and regularization parameter.

6. **Gradient Boosted Trees (GBT)**:
   - Training the Gradient Boosted Trees model to build an ensemble of weak learners.
   - Specify parameters such as the learning rate and maximum depth of trees.

7. **Multilayer Perceptron (MLP)**:
   - Training the Multilayer Perceptron model, a type of neural network.
   - Configure parameters such as the number of layers, neurons per layer, and activation function.

Ensure that you have preprocessed the data and selected relevant features before training the models.

## Evaluation Metrics

After training the models, evaluate their performance using the following metrics:

1. **Accuracy**:
   - Accuracy measures the ratio of correctly predicted instances to the total instances. It indicates the overall correctness of the model's predictions.

2. **Precision**:
   - Precision measures the ratio of correctly predicted positive observations to the total predicted positive observations. It indicates the accuracy of positive predictions.

3. **Recall**:
   - Recall (also known as sensitivity) measures the ratio of correctly predicted positive observations to the all observations in actual class. It indicates the model's ability to find all positive instances.

4. **F1 Score**:
   - F1 Score is the harmonic mean of precision and recall. It provides a balance between precision and recall, considering both false positives and false negatives.

Evaluating each model using these metrics to gain insights into their performance and choosing the best-performing model for the task. Ensure to validate the results on both training and testing datasets to assess the model's generalization ability.


## The Results

Here are the results of model evaluation:

| Model                   | Accuracy      | Precision     | Recall        | F1 Score      |
|-------------------------|---------------|---------------|---------------|---------------|
| Logistic Regression     | 0.85795       | 0.85819       | 0.85795       | 0.85804       |
| Decision Tree           | 0.99432       | 0.99439       | 0.99432       | 0.99432       |
| Random Forest           | 0.95455       | 0.95798       | 0.95455       | 0.95421       |
| Naïve Bayes             | 0.84659       | 0.84684       | 0.84659       | 0.84669       |
| Support Vector Machines | 0.85795       | 0.86151       | 0.85795       | 0.85837       |
| Gradient Boosted Trees | 0.99432       | 0.99439       | 0.99432       | 0.99432       |
| Multilayer Perceptron   | 0.97159       | 0.97162       | 0.97159       | 0.97157       |

These results demonstrate the performance of each model based on accuracy, precision, recall, and F1 score metrics.

## Conclusion and future enhancement

## Code Overview
