import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Object containing methods for loading data.
 */
object DataLoader {
  private val logger = LoggerFactory.getLogger(DataLoader.getClass)

  /**
   * Loads a CSV file into a DataFrame.
   *
   * @param spark SparkSession
   * @param csvFilePath Path to the CSV file
   * @return DataFrame containing the data from the CSV file
   */
  private def loadCSV(spark: SparkSession, csvFilePath: String): DataFrame = {
    logger.info(s"Loading CSV file from: $csvFilePath")
    try {
      spark.read
        .option("header", "true")
        .csv(csvFilePath)
    } catch {
      case e: Exception =>
        logger.error(s"Error loading CSV file from: $csvFilePath", e)
        throw e
    }
  }

  /**
   * Loads data from the specified file path, removing unnecessary columns.
   *
   * @param spark SparkSession
   * @param filePath Path to the data file
   * @return DataFrame containing the cleaned data
   */
  def loadData(spark: SparkSession, filePath: String): DataFrame = {
    logger.info(s"Loading data from: $filePath")
    val data = loadCSV(spark, filePath)
    logger.info("Removing columns: Project, Case_ID")
    data.drop("Project", "Case_ID")
  }
}
