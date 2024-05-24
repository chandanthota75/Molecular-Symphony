import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Class for training and evaluating a Logistic Regression model.
 */
class LRModel {

  private val logger = LoggerFactory.getLogger(classOf[LRModel])

  /**
   * Trains a Logistic Regression model and evaluates it on testing data.
   *
   * @param trainingData      DataFrame containing the training data
   * @param testingData       DataFrame containing the testing data
   * @param spark             SparkSession
   * @param labelCol          Name of the label column
   * @param featuresCol       Name of the features column
   * @param maxIter           Maximum number of iterations
   * @param regParam          Regularization parameter
   * @param elasticNetParam   Elastic net parameter
   * @param tol               Convergence tolerance
   * @param fitIntercept      Whether to fit an intercept term
   * @param standardization   Whether to standardize the features
   * @param threshold         Threshold in binary classification
   * @param weightCol         Optional weight column
   * @param aggregationDepth  Aggregation depth for treeAggregate
   * @param family            The name of family which is a description of the label distribution to be used in the model
   * @return Tuple containing the trained Logistic Regression model and DataFrame with predictions
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "Label",
                  featuresCol: String = "selectedFeatures",
                  maxIter: Int = 100,
                  regParam: Double = 0.0,
                  elasticNetParam: Double = 0.0,
                  tol: Double = 1e-6,
                  fitIntercept: Boolean = true,
                  standardization: Boolean = true,
                  threshold: Double = 0.5,
                  weightCol: Option[String] = None,
                  aggregationDepth: Int = 2,
                  family: String = "auto"
                ): (LogisticRegressionModel, DataFrame) = {

    try {
      logger.info("Training Logistic Regression model")

      // Creating a Logistic Regression instance with specified parameters
      val lr = new LogisticRegression()
        .setLabelCol(labelCol)
        .setFeaturesCol(featuresCol)
        .setMaxIter(maxIter)
        .setRegParam(regParam)
        .setElasticNetParam(elasticNetParam)
        .setTol(tol)
        .setFitIntercept(fitIntercept)
        .setStandardization(standardization)
        .setThreshold(threshold)
        .setAggregationDepth(aggregationDepth)
        .setFamily(family)

      // Fitting the Logistic Regression model to the training data
      val model = lr.fit(trainingData)

      // Making predictions on the testing data
      logger.info("Making predictions on testing data")
      val predictions = model.transform(testingData)

      // Calculating and save classification metrics
      logger.info("Calculating and saving classification metrics")
      ClassificationMetrics.calculateAndSaveMetrics(
        predictions,
        "Logistic Regression",
        "C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\LRModel\\results",
        spark
      )

      // Saving the trained Logistic Regression model
      logger.info("Saving Logistic Regression model")
      model.save("C:\\Users\\chand\\Desktop\\MolecularSymphony\\Models\\LRModel\\model")

      // Returning the trained model and predictions DataFrame
      (model, predictions)
    } catch {
      case e: Exception =>
        logger.error("Error occurred during model training", e)
        throw e
    }
  }
}
