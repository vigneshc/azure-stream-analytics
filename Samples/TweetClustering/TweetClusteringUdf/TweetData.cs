using System;
using Microsoft.ML.Data;

namespace TweetClusteringUdf
{
    internal class TweetData
    {
        [LoadColumn(0)]
        public string Text;
    }

    internal class TweetCluster
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}
