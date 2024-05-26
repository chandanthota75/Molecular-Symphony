import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Class for training and evaluating a Multilayer Perceptron model.
 */
class MLPModel {

  private val logger = LoggerFactory.getLogger(classOf[MLPModel])

  /**
   * Trains a Multilayer Perceptron model and evaluates it on testing data.
   *
   * @param trainingData      DataFrame containing the training data
   * @param testingData       DataFrame containing the testing data
   * @param spark             SparkSession
   * @param labelCol          Name of the label column
   * @param featuresCol       Name of the features column
   * @param layers            Array specifying the layer sizes including input size and output size
   * @param maxIter           Maximum number of iterations
   * @param blockSize         Size of blocks for blockwise parallelization
   * @param seed              Random seed
   * @return Tuple containing the trained Multilayer Perceptron model and DataFrame with predictions
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "Label",
                  featuresCol: String = "selectedFeatures",
                  layers: Array[Int],
                  maxIter: Int = 100,
                  blockSize: Int = 128,
                  seed: Long = 1234L
                ): (MultilayerPerceptronClassificationModel, DataFrame) = {

    try {
      logger.info("Training Multilayer Perceptron model")

      // Creating a Multilayer Perceptron instance with specified parameters
      val mlp = new MultilayerPerceptronClassifier()
        .setLabelCol(labelCol)
        .setFeaturesCol(featuresCol)
        .setLayers(layers)
        .setMaxIter(maxIter)
        .setBlockSize(blockSize)
        .setSeed(seed)

      // Fitting the Multilayer Perceptron model to the training data
      val model = mlp.fit(trainingData)

      // Making predictions on the testing data
      logger.info("Making predictions on testing data")
      val predictions = model.transform(testingData)

      // Calculating and saving classification metrics
      logger.info("Calculating and saving classification metrics")
      ClassificationMetrics.calculateAndSaveMetrics(
        predictions,
        "Multilayer Perceptron",
        "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\MLPModel\\results",
        spark
      )

      // Saving the Multilayer Perceptron model predictions
      DataFrameUtils.saveDataFrame(predictions.select(labelCol, "Prediction"), "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\MLPModel\\predictions")

      // Saving the trained Multilayer Perceptron model
      logger.info("Saving Multilayer Perceptron model")
      model.save("C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\MLPModel\\model")

      // Returning the trained model and predictions DataFrame
      (model, predictions)
    } catch {
      case e: Exception =>
        logger.error("Error occurred during model training", e)
        throw e
    }
  }
}
