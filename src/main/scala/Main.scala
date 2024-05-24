import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import DataLoader.loadData
import DataPreprocessing.processData

object Main {
  def main(args: Array[String]): Unit = {
    val logger = LoggerFactory.getLogger(getClass)

    System.setProperty("hadoop.home.dir", "C:/Users/chand/Desktop/MolecularSymphony/Hadoop")

    val spark = SparkSession.builder()
      .appName("MolecularSymphony")
      .master("local[*]")
      .getOrCreate()

    try {
      logger.info("Loading data from CSV")
      val dataframe = loadData(spark, "orgData/TCGA_GBM_LGG_Mutations_all.csv")

      logger.info("Processing data")
      val (train, test) = processData(spark, dataframe)

      logger.info("Training and Evaluating Logistic Regression model")
      new LRModel().trainModel(train, test, spark)

      logger.info("Training and Evaluating Decision Tree model")
      new DTModel().trainModel(train, test, spark)

      logger.info("Training and Evaluating Random Forest model")
      new RFModel().trainModel(train, test, spark)

      logger.info("Training and Evaluating Naive Bayes model")
      new NBModel().trainModel(train, test, spark)

      logger.info("Training and Evaluating Support Vector Classifier model")
      new SVCModel().trainModel(train, test, spark)

      logger.info("Training and Evaluating Gradient Boosted Trees model")
      new GBTModel().trainModel(train, test, spark)

      logger.info("Training and Evaluating Multilayer Perceptron model")
      new MLPModel().trainModel(train, test, spark, layers = Array(16, 128, 24, 2))

      logger.info("Model training and evaluation completed successfully")
    } catch {
      case e: java.io.FileNotFoundException =>
        logger.error(s"File not found: ${e.getMessage}")
      case e: org.apache.spark.SparkException =>
        logger.error(s"Spark error: ${e.getMessage}")
      case e: Exception =>
        logger.error(s"An error occurred: ${e.getMessage}", e)
    } finally {
      logger.info("Stopping Spark session")
      spark.stop()
    }
  }
}
