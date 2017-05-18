## Background

Hadoop is an open source version inspired by Google MapReduce and Google File System and designed for distributed processing of large data sets across a cluster of systems. Hadoop is designed with an assumption that all hardware fails sooner or later and that the system should be robust and able to handle the hardware failures automatically. This is why hadoop is considered to be <b>fault tolerant</b>, since the goal is to have nodes fail without suffering any loss of data.

Apache Hadoop consists of two core components:


### HDFS

The Hadoop Distributed File System is a filesystem that manages the storage across a network of machines. This is an incredibly complexity problem to handle without the use of big data technology like hadoop. 

### MapReduce

Framework and API for MapReduce jobs. MapReduce is a software framework or programming model, which enable users to write programs so that data can be processed parallelly across multiple systems in a cluster. MapReduce consists of two parts: 


#### Map 

Map task is performed using a `map()` function that basically performs filtering and sorting. This part is responsible for processing one or more chunks of data and producing the output results which are generally referred as intermediate results. As shown in the diagram below, map task is generally processed in parallel provided the maping operation is independent of each other.


#### Reduce

Reduce task is performed by `reduce()` function and performs a summary operation. It is responsible for consolidating the results produced by each of the Map task.


### Processing Patterns

There are different ways in which hadoop processes its data, four of which are overviewed below:

1. Batch processing: Batch Processing is Hadoop's classic feature and is specifically when the processing job is known and ready to be run over large amounts of data.

2. Interactive SQL: 

3. Iterative processing: Many algorithms - such as those in machine learning - are iterative in nature, so it’s much more efficient to hold each intermediate working set in memory, compared to loading from disk on each iteration. The architecture of MapReduce does not allow this, but it’s straightforward with Spark, for example, and it enables a highly exploratory style of working with datasets.

4. Stream processing: Streaming systems like Storm, Spark Streaming, or Samza make it possible to run real-time, distributed computations on unbounded streams of data and emit results to Hadoop storage or external systems.
