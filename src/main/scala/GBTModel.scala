import org.apache.spark.ml.classification.{GBTClassifier, GBTClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Class for training and evaluating a Gradient Boosting Trees model.
 */
class GBTModel {

  private val logger = LoggerFactory.getLogger(classOf[GBTModel])

  /**
   * Trains a Gradient Boosting Trees model and evaluates it on testing data.
   *
   * @param trainingData      DataFrame containing the training data
   * @param testingData       DataFrame containing the testing data
   * @param spark             SparkSession
   * @param labelCol          Name of the label column
   * @param featuresCol       Name of the features column
   * @param maxDepth          Maximum depth of the tree
   * @param maxBins           Maximum number of bins used for discretizing continuous features
   * @param maxIter           Maximum number of iterations
   * @param stepSize          Step size for the initial learning rate
   * @return Tuple containing the trained Gradient Boosting Trees model and DataFrame with predictions
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "Label",
                  featuresCol: String = "selectedFeatures",
                  maxDepth: Int = 5,
                  maxBins: Int = 32,
                  maxIter: Int = 10,
                  stepSize: Double = 0.1
                ): (GBTClassificationModel, DataFrame) = {

    try {
      logger.info("Training Gradient Boosting Trees model")

      // Creating a Gradient Boosting Trees instance with specified parameters
      val gbt = new GBTClassifier()
        .setLabelCol(labelCol)
        .setFeaturesCol(featuresCol)
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
        .setMaxIter(maxIter)
        .setStepSize(stepSize)

      // Fitting the Gradient Boosting Trees model to the training data
      val model = gbt.fit(trainingData)

      // Making predictions on the testing data
      logger.info("Making predictions on testing data")
      val predictions = model.transform(testingData)

      // Calculating and saving classification metrics
      logger.info("Calculating and saving classification metrics")
      ClassificationMetrics.calculateAndSaveMetrics(
        predictions,
        "Gradient Boosting Trees",
        "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\GBTModel\\results",
        spark
      )

      // Saving the Gradient Boosting Trees model predictions
      DataFrameUtils.saveDataFrame(predictions.select(labelCol, "Prediction"), "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\GBTModel\\predictions")

      // Saving the trained Gradient Boosting Trees model
      logger.info("Saving Gradient Boosting Trees model")
      model.save("C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\GBTModel\\model")

      // Returning the trained model and predictions DataFrame
      (model, predictions)
    } catch {
      case e: Exception =>
        logger.error("Error occurred during model training", e)
        throw e
    }
  }
}
