import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Class for training and evaluating a Decision Tree model.
 */
class DTModel {

  private val logger = LoggerFactory.getLogger(classOf[DTModel])

  /**
   * Trains a Decision Tree model and evaluates it on testing data.
   *
   * @param trainingData          DataFrame containing the training data
   * @param testingData           DataFrame containing the testing data
   * @param spark                 SparkSession
   * @param labelCol              Name of the label column
   * @param featuresCol           Name of the features column
   * @param maxDepth              Maximum depth of the tree
   * @param maxBins               Maximum number of bins used for discretizing continuous features
   * @param minInstancesPerNode   Minimum number of instances each child must have after split
   * @param minInfoGain           Minimum information gain required for a split
   * @return Tuple containing the trained Decision Tree model and DataFrame with predictions
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "Label",
                  featuresCol: String = "selectedFeatures",
                  maxDepth: Int = 5,
                  maxBins: Int = 32,
                  minInstancesPerNode: Int = 1,
                  minInfoGain: Double = 0.0
                ): (DecisionTreeClassificationModel, DataFrame) = {

    try {
      logger.info("Training Decision Tree model")

      // Creating a Decision Tree instance with specified parameters
      val dt = new DecisionTreeClassifier()
        .setLabelCol(labelCol)
        .setFeaturesCol(featuresCol)
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
        .setSeed(1234L)
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)

      // Fitting the Decision Tree model to the training data
      val model = dt.fit(trainingData)

      // Making the predictions on the testing data
      logger.info("Making predictions on testing data")
      val predictions = model.transform(testingData)

      // Calculating and save classification metrics
      logger.info("Calculating and saving classification metrics")
      ClassificationMetrics.calculateAndSaveMetrics(
        predictions,
        "Decision Tree",
        "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\DTModel\\results",
        spark
      )

      // Saving the Decision Tree model predictions
      DataFrameUtils.saveDataFrame(predictions.select(labelCol, "Prediction"), "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\DTModel\\predictions")

      // Saving the trained Decision Tree model
      logger.info("Saving Decision Tree model")
      model.save("C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\DTModel\\model")

      // Returning the trained model and predictions DataFrame
      (model, predictions)
    } catch {
      case e: Exception =>
        logger.error("Error occurred during model training", e)
        throw e
    }
  }
}
