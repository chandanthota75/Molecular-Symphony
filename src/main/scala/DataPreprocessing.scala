import org.apache.spark.ml.feature.{UnivariateFeatureSelector, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, rand, udf}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.slf4j.LoggerFactory

/**
 * Object containing methods for data preprocessing.
 */
object DataPreprocessing {
  private val logger = LoggerFactory.getLogger(DataPreprocessing.getClass)

  // Defining schema for the DataFrame
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

  /**
   * Shuffles the given DataFrame.
   *
   * @param df DataFrame to be shuffled
   * @return Shuffled DataFrame
   */
  private def shuffleDataFrame(df: DataFrame): DataFrame = df.orderBy(rand())

  /**
   * Processes the data by cleaning, mapping, transforming, and splitting it into train and test sets.
   *
   * @param spark SparkSession
   * @param df DataFrame to be processed
   * @return Tuple containing the training DataFrame and the testing DataFrame
   */
  def processData(spark: SparkSession, df: DataFrame): (DataFrame, DataFrame) = {
    import spark.implicits._

    // UDF to extract age in years from a string
    val extractAgeInYearsUDF = udf((ageString: String) => {
      val agePattern = "(\\d+) years (\\d+) days".r
      ageString match {
        case agePattern(years, days) =>
          val ageDecimal = years.toInt + (days.toDouble / 365.0)
          BigDecimal(ageDecimal).setScale(1, BigDecimal.RoundingMode.HALF_UP).toDouble
        case _ => 0.0
      }
    })

    // Mappings for categorical data
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

    // UDFs for mapping categorical data
    val genderUDF = udf((gender: String) => genderMapping.getOrElse(gender, -1))
    val raceUDF = udf((race: String) => raceMapping.getOrElse(race.toLowerCase, -1))
    val primaryDiagnosisUDF = udf((diagnosis: String) => primaryDiagnosisMapping.getOrElse(diagnosis, -1))
    val mutationUDF = udf((mutation: String) => mutationMapping.getOrElse(mutation, -1))
    val gradeUDF = udf((grade: String) => gradeMapping.getOrElse(grade, -1))

    try {
      logger.info("Dropping rows with null values")
      val cleanedDF = df.na.drop()

      logger.info("Mapping and transforming columns")
      val mappedDF = cleanedDF
        .withColumn("Age_at_diagnosis", extractAgeInYearsUDF($"Age_at_diagnosis"))
        .withColumn("Gender", genderUDF(col("Gender")))
        .withColumn("Primary_Diagnosis", primaryDiagnosisUDF(col("Primary_Diagnosis")))
        .withColumn("Race", raceUDF(col("Race")))
        .withColumn("IDH1", mutationUDF(col("IDH1")))
        .withColumn("TP53", mutationUDF(col("TP53")))
        .withColumn("ATRX", mutationUDF(col("ATRX")))
        .withColumn("PTEN", mutationUDF(col("PTEN")))
        .withColumn("EGFR", mutationUDF(col("EGFR")))
        .withColumn("CIC", mutationUDF(col("CIC")))
        .withColumn("MUC16", mutationUDF(col("MUC16")))
        .withColumn("PIK3CA", mutationUDF(col("PIK3CA")))
        .withColumn("NF1", mutationUDF(col("NF1")))
        .withColumn("PIK3R1", mutationUDF(col("PIK3R1")))
        .withColumn("FUBP1", mutationUDF(col("FUBP1")))
        .withColumn("RB1", mutationUDF(col("RB1")))
        .withColumn("NOTCH1", mutationUDF(col("NOTCH1")))
        .withColumn("BCOR", mutationUDF(col("BCOR")))
        .withColumn("CSMD3", mutationUDF(col("CSMD3")))
        .withColumn("SMARCA4", mutationUDF(col("SMARCA4")))
        .withColumn("GRIN2A", mutationUDF(col("GRIN2A")))
        .withColumn("IDH2", mutationUDF(col("IDH2")))
        .withColumn("FAT4", mutationUDF(col("FAT4")))
        .withColumn("PDGFRA", mutationUDF(col("PDGFRA")))
        .withColumn("Label", gradeUDF(col("Grade")))

      logger.info("Selecting and shuffling final DataFrame")
      val finalDF = mappedDF.selectExpr(schema.fieldNames.map(fieldName => s"`$fieldName`"): _*)
      val shuffledDF = shuffleDataFrame(finalDF)

      // Saving the processed DataFrame
      DataFrameUtils.saveDataFrame(shuffledDF, "proData/processed")

      // Assembling feature vectors
      val featureCols = schema.fieldNames.filterNot(_ == "Label")
      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("features")

      logger.info("Assembling feature vectors")
      val assembledDF = assembler.transform(shuffledDF).select("features", "Label")

      // Performing feature selection
      logger.info("Performing feature selection using UnivariateFeatureSelector")
      val selector = new UnivariateFeatureSelector()
        .setSelectionMode("numTopFeatures")
        .setSelectionThreshold(16)
        .setFeatureType("categorical")
        .setLabelType("categorical")
        .setFeaturesCol("features")
        .setLabelCol("Label")
        .setOutputCol("selectedFeatures")

      val selectedDF = selector.fit(assembledDF).transform(assembledDF).select("selectedFeatures", "Label")

      // Splitting the data into training and testing datasets
      logger.info("Splitting data into train and test sets")
      val Array(trainDF, testDF) = selectedDF.randomSplit(Array(0.8, 0.2))

      // Saving train and test datasets to CSV
      logger.info("Saving train and test datasets to CSV")
      saveDataFrameAsCSV(trainDF, "proData/train")
      saveDataFrameAsCSV(testDF, "proData/test")

      (trainDF, testDF)
    } catch {
      case e: Exception =>
        logger.error("An error occurred during data preprocessing", e)
        throw e
    }
  }

  /**
   * Saves a DataFrame as a CSV file.
   *
   * @param df DataFrame to be saved
   * @param path Path to save the CSV file
   */
  private def saveDataFrameAsCSV(df: DataFrame, path: String): Unit = {
    // UDF to convert a Vector to a comma-separated string
    val toStringUDF = udf((vector: Vector) => vector.toArray.mkString(","))
    df.withColumn("selectedFeatures", toStringUDF(col("selectedFeatures")))
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv(path)
  }
}
