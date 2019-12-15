using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace TweetClusteringUdf
{
    public static class TweetClustering
    {
        static Dictionary<long, PredictionEngine<TweetData, TweetCluster>> predictionEngines = new Dictionary<long, PredictionEngine<TweetData, TweetCluster>>();

        // given a collection of strings and cluster size, returns a KMeans clustering model encoded as base64 string.
        public static string GetClusteringModel(string[] data, int clusterSize)
        {
            var mlContext = new MLContext(seed: 15);
            IDataView tweetDataView = mlContext.Data.LoadFromEnumerable<TweetData>(data.Select(text => new TweetData() { Text = text}));

            var clusterPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text")
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("TokensWithoutStopWords","Tokens"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("TokensWithoutStopWords"))
                .Append(
                    mlContext.Transforms.Text.ProduceHashedNgrams(
                        "NgramFeatures",
                        "TokensWithoutStopWords",
                        numberOfBits: 5,
                        ngramLength: 3,
                        useAllLengths: false,
                        maximumNumberOfInverts: 1))
                .Append(mlContext.Clustering.Trainers.KMeans("NgramFeatures", numberOfClusters: clusterSize));

            var model = clusterPipeline.Fit(tweetDataView);
            using(var ms = new MemoryStream())
            {
                mlContext.Model.Save(model, tweetDataView.Schema, ms);
                return Convert.ToBase64String(ms.ToArray());
            }
        }

        // given a modelVersion, base64 encoded model and a text, creates a model and returns predicted cluster.
        public static long PredictCluster(string text, long modelVersion, string base64ModelString)
        {
            return GetPredictor(modelVersion, base64ModelString)
            .Predict(new TweetData() { Text = text })
            .PredictedClusterId;
        }

        private static PredictionEngine<TweetData, TweetCluster> GetPredictor(long modelVersion, string base64ModelString)
        {
            if(predictionEngines.TryGetValue(modelVersion, out PredictionEngine<TweetData, TweetCluster> predictor))
            {
                return predictor;
            }

            using(var ms = new MemoryStream(Convert.FromBase64String(base64ModelString)))
            {
                var mlContext = new MLContext(seed: 15);
                var model = mlContext.Model.Load(ms, out _);
                predictor = mlContext.Model.CreatePredictionEngine<TweetData, TweetCluster>(model);
                predictionEngines[modelVersion] = predictor;
                return predictor;
            }
        }
    }
}