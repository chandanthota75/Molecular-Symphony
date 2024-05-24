import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Class for training and evaluating a Naive Bayes model.
 */
class NBModel {

  private val logger = LoggerFactory.getLogger(classOf[NBModel])

  /**
   * Trains a Naive Bayes model and evaluates it on testing data.
   *
   * @param trainingData DataFrame containing the training data
   * @param testingData DataFrame containing the testing data
   * @param spark SparkSession
   * @param labelCol Name of the label column
   * @param featuresCol Name of the features column
   * @param smoothing Smoothing parameter
   * @param modelType Model type (multinomial or bernoulli)
   * @param thresholds Optional thresholds for binary classification
   * @param weightCol Optional column for instance weights
   * @return Tuple containing the trained Naive Bayes model and DataFrame with predictions
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "Label",
                  featuresCol: String = "selectedFeatures",
                  smoothing: Double = 1.0,
                  modelType: String = "multinomial",
                  thresholds: Option[Array[Double]] = None,
                  weightCol: Option[String] = None
                ): (NaiveBayesModel, DataFrame) = {

    try {
      logger.info("Training Naive Bayes model")

      // Creating a Naive Bayes instance with specified parameters
      val nb = new NaiveBayes()
        .setLabelCol(labelCol)
        .setFeaturesCol(featuresCol)
        .setSmoothing(smoothing)
        .setModelType(modelType)

      // Fitting the Naive Bayes model to the training data
      val model = nb.fit(trainingData)

      // Making predictions on the testing data
      logger.info("Making predictions on testing data")
      val predictions = model.transform(testingData)

      // Calculating and saving classification metrics
      logger.info("Calculating and saving classification metrics")
      ClassificationMetrics.calculateAndSaveMetrics(
        predictions,
        "Naive Bayes",
        "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\NBModel\\results",
        spark
      )

      // Saving the trained Naive Bayes model
      logger.info("Saving Naive Bayes model")
      model.save("C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\NBModel\\model")

      // Returning the trained model and predictions DataFrame
      (model, predictions)
    } catch {
      case e: Exception =>
        logger.error("Error occurred during model training", e)
        throw e
    }
  }
}
