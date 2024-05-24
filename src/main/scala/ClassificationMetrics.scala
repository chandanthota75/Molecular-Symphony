import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.lit
import org.slf4j.LoggerFactory

/**
 * Object for calculating and saving classification metrics.
 */
object ClassificationMetrics {
  private val logger = LoggerFactory.getLogger(ClassificationMetrics.getClass)

  /**
   * Calculates and saves classification metrics for a given model.
   *
   * @param predictions DataFrame containing the predictions made by the model
   * @param modelName Name of the model
   * @param outputPath Output path to save the metrics
   * @param spark SparkSession
   */
  def calculateAndSaveMetrics(
                               predictions: DataFrame,
                               modelName: String,
                               outputPath: String,
                               spark: SparkSession
                             ): Unit = {
    try {
      logger.info(s"Calculating and saving metrics for model: $modelName")
      import spark.implicits._

      // Creating a MulticlassClassificationEvaluator to calculate metrics
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("Label")
        .setPredictionCol("prediction")

      // Calculating accuracy, precision, recall, and F1 score
      val metrics = Seq(
        ("Accuracy", evaluator.setMetricName("accuracy").evaluate(predictions)),
        ("Precision", evaluator.setMetricName("weightedPrecision").evaluate(predictions)),
        ("Recall", evaluator.setMetricName("weightedRecall").evaluate(predictions)),
        ("F1 Score", evaluator.setMetricName("f1").evaluate(predictions))
      )

      // Converting metrics to DataFrame for easy saving
      val metricsDF = metrics.toDF("Metric", "Value").withColumn("Model", lit(modelName))

      logger.info(s"Saving metrics to: $outputPath")
      // Saving metrics DataFrame to the specified output path
      DataFrameUtils.saveDataFrame(metricsDF, outputPath)
    } catch {
      case e: Exception =>
        // Logging and rethrow any exceptions that occur during metric calculation
        logger.error(s"Error calculating and saving metrics for model: $modelName", e)
        throw e
    }
  }
}
