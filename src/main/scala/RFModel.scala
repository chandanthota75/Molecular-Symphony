import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Class for training and evaluating a Random Forest model.
 */
class RFModel {

  private val logger = LoggerFactory.getLogger(classOf[RFModel])

  /**
   * Trains a Random Forest model and evaluates it on testing data.
   *
   * @param trainingData DataFrame containing the training data
   * @param testingData DataFrame containing the testing data
   * @param spark SparkSession
   * @param labelCol Name of the label column
   * @param featuresCol Name of the features column
   * @param numTrees Number of trees in the forest
   * @param maxDepth Maximum depth of the tree
   * @param maxBins Maximum number of bins used for discretizing continuous features
   * @param minInstancesPerNode Minimum number of instances each child must have after split
   * @param minInfoGain Minimum information gain for a split to be considered at a node
   * @param subsamplingRate Fraction of the training data used for learning each decision tree
   * @param featureSubsetStrategy Strategy for selecting a subset of features at each node
   * @return Tuple containing the trained Random Forest model and DataFrame with predictions
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "Label",
                  featuresCol: String = "selectedFeatures",
                  numTrees: Int = 20,
                  maxDepth: Int = 5,
                  maxBins: Int = 32,
                  minInstancesPerNode: Int = 1,
                  minInfoGain: Double = 0.0,
                  subsamplingRate: Double = 1.0,
                  featureSubsetStrategy: String = "auto"
                ): (RandomForestClassificationModel, DataFrame) = {

    try {
      logger.info("Training Random Forest model")

      // Creating a Random Forest instance with specified parameters
      val rf = new RandomForestClassifier()
        .setLabelCol(labelCol)
        .setFeaturesCol(featuresCol)
        .setNumTrees(numTrees)
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)
        .setSubsamplingRate(subsamplingRate)
        .setFeatureSubsetStrategy(featureSubsetStrategy)

      // Fitting the Random Forest model to the training data
      val model = rf.fit(trainingData)

      // Making predictions on the testing data
      logger.info("Making predictions on testing data")
      val predictions = model.transform(testingData)

      // Calculating and saving classification metrics
      logger.info("Calculating and saving classification metrics")
      ClassificationMetrics.calculateAndSaveMetrics(
        predictions,
        "Random Forest",
        "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\RFModel\\results",
        spark
      )

      // Saving the Random Forest model predictions
      DataFrameUtils.saveDataFrame(predictions.select(labelCol, "Prediction"), "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\RFModel\\predictions")

      // Saving the trained Random Forest model
      logger.info("Saving Random Forest model")
      model.save("C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\RFModel\\model")

      // Returning the trained model and predictions DataFrame
      (model, predictions)
    } catch {
      case e: Exception =>
        logger.error("Error occurred during model training", e)
        throw e
    }
  }
}
