using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using TweetClusteringUdf;

namespace Runner
{
    // Runner to test changes in Tweet clustering udf locally.
    class Program
    {
        static void Main(string[] args)
        {
            switch(args[0])
            {
                case "saveModel":
                {
                    // given an input file, and model location, saves a model that clusters input tweets into 5 clusters.
                    var inputFileName = args[1];
                    var tweetData = GetTweetTexts(inputFileName)
                        .ToArray();
                    
                    var modelString = TweetClusteringUdf.TweetClustering.GetClusteringModel(tweetData, clusterSize: 5);
                    File.WriteAllText(args[2], modelString);
                    break;
                }
                case "score":
                {
                    // given an input file and model location, prints predicted cluster for each tweet.
                    var inputFileName = args[1];
                    var modelFile = args[2];
                    var modelText = File.ReadAllText(modelFile);

                    foreach(var t in GetTweetTexts(inputFileName))
                    {
                        Console.WriteLine($"{TweetClusteringUdf.TweetClustering.PredictCluster(t, 1, modelText)} === {t}");
                    }
                    
                    break;
                }
                default:
                    Console.WriteLine($"Unknown option: {args[0]}");
                    break;
            }
        }

        static IEnumerable<string> GetTweetTexts(string inputFileName)
        {
            var tweetData = File.ReadAllLines(inputFileName)
                .Select(l => JsonConvert.DeserializeObject<JObject>(l))
                .Select(j => j.Value<string>("text"));
            return tweetData;
        }
    }
}
