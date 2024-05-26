import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Class for training and evaluating a Support Vector Machine model.
 */
class SVCModel {

  private val logger = LoggerFactory.getLogger(classOf[SVCModel])

  /**
   * Trains a Support Vector Machine model and evaluates it on testing data.
   *
   * @param trainingData DataFrame containing the training data
   * @param testingData DataFrame containing the testing data
   * @param spark SparkSession
   * @param labelCol Name of the label column
   * @param featuresCol Name of the features column
   * @param maxIter Maximum number of iterations
   * @param regParam Regularization parameter
   * @param tol Convergence tolerance for iterative algorithms
   * @param fitIntercept Whether to fit an intercept term
   * @param standardization Whether to standardize the training features before fitting the model
   * @param threshold Threshold in binary classification prediction
   * @param weightCol Optional column for weights
   * @param aggregationDepth Suggested depth for tree aggregation
   * @return Tuple containing the trained Linear SVC model and DataFrame with predictions
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "Label",
                  featuresCol: String = "selectedFeatures",
                  maxIter: Int = 100,
                  regParam: Double = 0.0,
                  tol: Double = 1e-6,
                  fitIntercept: Boolean = true,
                  standardization: Boolean = true,
                  threshold: Double = 0.0,
                  weightCol: Option[String] = None,
                  aggregationDepth: Int = 2
                ): (LinearSVCModel, DataFrame) = {

    try {
      logger.info("Training Support Vector Machine model")

      // Creating a Linear SVC instance with specified parameters
      val lsvc = new LinearSVC()
        .setLabelCol(labelCol)
        .setFeaturesCol(featuresCol)
        .setMaxIter(maxIter)
        .setRegParam(regParam)
        .setTol(tol)
        .setFitIntercept(fitIntercept)
        .setStandardization(standardization)
        .setThreshold(threshold)
        .setAggregationDepth(aggregationDepth)

      // Fitting the Linear SVC model to the training data
      val model = lsvc.fit(trainingData)

      // Making predictions on the testing data
      logger.info("Making predictions on testing data")
      val predictions = model.transform(testingData)

      // Calculating and save classification metrics
      logger.info("Calculating and saving classification metrics")
      ClassificationMetrics.calculateAndSaveMetrics(
        predictions,
        "Support Vector Machine",
        "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\SVCModel\\results",
        spark
      )

      // Saving the Linear SVC model predictions
      DataFrameUtils.saveDataFrame(predictions.select(labelCol, "Prediction"), "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\SVCModel\\predictions")

      // Saving the trained Linear SVC model
      logger.info("Saving Support Vector Machine model")
      model.save("C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\SVCModel\\model")

      // Returning the trained model and predictions DataFrame
      (model, predictions)
    } catch {
      case e: Exception =>
        logger.error("Error occurred during model training", e)
        throw e
    }
  }
}
