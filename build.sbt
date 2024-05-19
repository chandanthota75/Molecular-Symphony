ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    // Project name and IDE package prefix
    name := "MolecularSymphony",
  )

// Define versions for Spark and Hadoop
val spark_version = "3.5.0"
val hadoop_version = "3.3.5"

// Define library dependencies
libraryDependencies ++= Seq(
  // Spark dependencies
  "org.apache.spark" %% "spark-core" % spark_version,
  "org.apache.spark" %% "spark-sql" % spark_version,
  "org.apache.spark" %% "spark-mllib" % spark_version,

  // Hadoop dependencies
  "org.apache.hadoop" % "hadoop-common" % hadoop_version,
  "org.apache.hadoop" % "hadoop-client" % hadoop_version,
  "org.apache.hadoop" % "hadoop-hdfs" % hadoop_version
)