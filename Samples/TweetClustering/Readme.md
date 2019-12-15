Contains C# code for creating a KMeans clustering model that clusters text.
This UDF will be used in a stream analytics job that does the following

1. Groups tweets into 15 minute windows every 5 minutes and create a clustering model.
2. Uses the model to predict the clusters for next 5 minutes.
3. Uses the cluster number as one of the grouping key.

The example demonstrates the ability to use a ML model in stream analytics job and shows a pattern for doing training in the stream analytics job.