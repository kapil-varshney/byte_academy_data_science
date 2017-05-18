## 1.0 Background

Spark is a low-level system for distributed computation on clusters, capable of doing in-memory caching between stages, improving the performance. This is in contrast to Hadoop, which instead writes everything to disk. Spark is also a much more flexible system in that it's not only constrained to MapReduce. 

Now, it might seem as though Spark is a replacement for Hadoop; and though it sometimes is used as a replacement, it can also be used to complement Hadoop's functionality. By running Spark on top of a Hadoop cluster, you can still leverage HDFS and YARN and then have Spark replace MapReduce. 
