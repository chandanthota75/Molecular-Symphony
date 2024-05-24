import org.apache.spark.sql.DataFrame
import org.slf4j.LoggerFactory

/**
 * Utility object for DataFrame operations.
 */
object DataFrameUtils {
  private val logger = LoggerFactory.getLogger(DataFrameUtils.getClass)

  /**
   * Calculates the shape of a DataFrame.
   *
   * @param df DataFrame to calculate the shape of
   * @return A tuple containing the number of rows and the number of columns
   */
  def shape(df: DataFrame): (Long, Int) = {
    try {
      (df.count(), df.columns.length)
    } catch {
      case e: Exception =>
        logger.error("Failed to calculate DataFrame shape", e)
        throw e
    }
  }

  /**
   * Saves a DataFrame to a specified path in CSV format.
   *
   * @param df DataFrame to save
   * @param path Path where the DataFrame should be saved
   */
  def saveDataFrame(df: DataFrame, path: String): Unit = {
    try {
      df.write
        .format("csv")
        .option("header", "true")
        .mode("overwrite")
        .csv(path)
    } catch {
      case e: Exception =>
        logger.error(s"Failed to save DataFrame to CSV at: $path", e)
        throw e
    }
  }
}
