using Animus;
using Animus.ANN.Core;
using Animus.ANN.Optimizer.DataMining;
using Animus.ANN.Types;
using Animus.ANN.Types.Layers;
using Animus.Tensor;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace AnimusUnitTest.ANNTest
{
    [TestClass]
    public class DataminingTest
    {
        [TestMethod]
        public void TestCalculation()
        {
            var ff = new FeedForwardNet<int>
                (
                    inputLayer: LayerFactory.MakeInputLayer<int>(i => new Vector(i), 1),
                    hiddenLayers: new[] { 5, 8, 7 }.Select(LayerFactory.MakeTanhLayer),
                    outputLayer: LayerFactory.MakeSoftmaxLayer(4)
                );

            var p = new LayerParameterList(ff.Initialize());

            foreach (var item in p)
            {
                Console.WriteLine(item.Weights + "\n");
            }

            var f = ff.BuildFlow();

            var input = 5;

            var output1 = f.Train(p, input);
            var output2 = f.Model(p, input);

            var v = new Vector(5);

            var h = p.Take(3).Aggregate(v, (v1, lp) => TensorFunction.Tanh(lp.Weights * v1 + lp.Bias));

            var o = TensorFunction.Softmax(p[p.Count - 1].Weights * h + p[p.Count - 1].Bias);

            for (int i = 0; i < o.Length; i++)
            {
                Assert.AreEqual((double)o[i], (double)output1[i]);
                Assert.AreEqual((double)o[i], (double)output2[i]);
            }
        }

        [TestMethod]
        public async Task TestGradient()
        {
            var data = DownloadJson();

            var ff = new FeedForwardNet<Iris>(
                    inputLayer: LayerFactory.MakeInputLayer((Iris ir) => new Vector(ir.PetalLength, ir.PetalWidth, ir.SepalLength, ir.SepalWidth), 4),
                    hiddenLayers: new[] { 16, 8 }.Select(LayerFactory.MakeTanhLayer),
                    outputLayer: LayerFactory.MakeSoftmaxLayer(3));

            var repos = new IrisRepository(data);

            var optimizer = new Optimizer();

            var flow = ff.BuildFlow();

            var p = await optimizer.TrainAsync(repos, flow.Train, new LayerParameterList(ff.Initialize()));

            var model = await optimizer.Train(ff, repos);

            foreach (var iris in data.GroupBy(ir => ir.Type).SelectMany(g => Rng.SelectNoRepeat(g, 1)))
            {
                var output = model.Compute(iris);

                // check the output
                Console.WriteLine($"{iris.Type}: {output}");
            }
        }


        public async Task Unsupervised()
        {
            var encoder = new FeedForwardNet<double[]>(
                    inputLayer: LayerFactory.MakeInputLayer((double[] arr) => new Vector(arr), 20),
                    hiddenLayers: new[] { 64, 32, 16 }.Select(LayerFactory.MakeTanhLayer),
                    outputLayer: LayerFactory.MakeTanhLayer(4));

            var decoder = new FeedForwardNet<Vector>(
                    inputLayer: LayerFactory.MakeInputLayer((Vector v)=> v, 4),
                    hiddenLayers: new[] { 16, 32, 64 }.Select(LayerFactory.MakeTanhLayer),
                    outputLayer: LayerFactory.MakeSoftmaxLayer(20));

            var encodeFlow = encoder.BuildFlow();
            var decoderFlow = decoder.BuildFlow();

            Func<LayerParameterList, double[], Vector> fullFlow = (w, data) =>
            {
                var hash = encodeFlow.Train(w, data);

                var decodeData = decoderFlow.Train(new LayerParameterList(w, 4, null), hash);

                return decodeData;
            };

            var encodeParam = encoder.Initialize();
            var decoderParam = decoder.Initialize();

            var p = new LayerParameterList(encodeParam.Concat(decoderParam));

            IDataRepository<double[], Vector> repos = null;

            var optimizer = new Optimizer();

            var finalParam = await optimizer.TrainAsync(repos, fullFlow, p);

            #region Find clusters

            IEnumerable<double[]> data = null; // {{ a1, a2,.... an},{ a1, a2,.... am},{ a1, a2,.... at}....} //input TOTAL data[100]  :100 sessions

            var hashes = data.Select(d => encodeFlow.Model(finalParam, d)); // list of hashes

            // kmean hashes
            // for ex: we get out 3 central points (clusters) --> clusters[3]
            IEnumerable<Vector> clusters = fit(hashes); //  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html //install them : anaconda x64 + pycharm

            var originalData = clusters.Select(c =>(double[]) decoderFlow.Model(new LayerParameterList(finalParam, 4, null), c)); // list of histograms

            // pick 30 - label of each cluster
            // p = .....
            //IEnumerable<double[]> testdata;
            //int i = 0;
            //foreach(var c in clusters)
            //{
            //    testdata[i] = hashes(c);
            //    i++;
            //}

            IEnumerable<int>  predictedClusters = predict(hashes); // [0,1,2,0,2,1,1......] : 100 items (belongs to 3 central points, clusters)

            // --> pick 30 randomly of each clusters (3 clusters, central points) 
            // 30 of clusters[0] , etc... then give a decision humanly it's a robot case , if yes mark 1
            // then calculate p (Robot lower bounding), p^ = (count if 1 )/ n = 30 , ex: p = 0.87

            #endregion

            #region check new session 

            //IDictionary<int, double> prob = null;

            // real
            double[] session = null; // { a1, a2,.... an}  //TestData for checking robot or.. 

            // hash data
            var hash1 = encodeFlow.Model(finalParam, session); // encode

            IEnumerable<int> foundCluster = predict(new Vector[] { hash1 });

            //var distance = double.PositiveInfinity;
            //var center = 0;

            //foreach (var item in clusters)
            //{
            //    var d = (double) ((hash1 - item) * (hash1 - item));
            //    var v = (hash1 - item).Pow(2); //ví du
            //    if(d < distance)
            //    {
            //        distance = d;
            //        break;
            //    }

            //    center++;
            //}

            //var realProb = prob[center];

            #endregion

            // anomaly
            var reconstruction = fullFlow(finalParam, session);

            var error = (double)repos.Loss(session, reconstruction);

            var std = 0.23;// +- 0.23 , độ lệch chuẩn

            var th = 3 * std;// 0.7

            //var anomalyScore =
            //   ps
            //       .Select((p, i) => -p * Real.Log(qs[i]) - (1 - p) * Real.Log(1 - qs[i]))
            //       .Count(r => (double)r >= 0.7);

            // the empirical rule https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
        }

        public IEnumerable<Iris> DownloadJson()
        {
            using (var web = new WebClient())
            {
                return
                    web.DownloadString("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
                    .Split('\n')
                    .Where(str => !string.IsNullOrEmpty(str))
                    .Select(str => str.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
                    .Select(arr => new Iris
                    {
                        SepalLength = Convert.ToDouble(arr[0]),
                        SepalWidth = Convert.ToDouble(arr[1]),
                        PetalLength = Convert.ToDouble(arr[2]),
                        PetalWidth = Convert.ToDouble(arr[3]),
                        Type = arr[4]
                    }).ToArray();
            }
        }
    }

    public class Iris
    {
        public double SepalLength { get; set; }

        public double SepalWidth { get; set; }

        public double PetalLength { get; set; }

        public double PetalWidth { get; set; }

        public string Type { get; set; }
    }

    public class IrisRepository : IDataRepository<Iris, Vector>
    {
        private readonly IEnumerable<Iris> data;

        public IrisRepository(IEnumerable<Iris> data)
        {
            this.data = data;
        }

        public Scalar Loss(Iris sample, Vector output)
        {
            switch (sample.Type)
            {
                case "Iris-setosa":
                    return -TensorFunction.Log(output[0]);

                case "Iris-versicolor":
                    return -TensorFunction.Log(output[1]);

                case "Iris-virginica":
                    return -TensorFunction.Log(output[2]);

                default:
                    throw new Exception("Unknown iris");
            }
        }

        public IEnumerable<Iris> Sample() => Rng.SelectNoRepeat(data, 30);
    }

    public class FPSRepos : IDataRepository<double[], Vector>
    {
        public Scalar Loss(double[] sample, Vector output)
        {
            return -new Vector(sample) * TensorFunction.Log(output);
        }

        public IEnumerable<double[]> Sample() => null;//db;
    }

    public class Optimizer : Sgd
    {
        protected override double LearningRate(int epoch) => 0.001;

        protected override double Momentum(int epoch) => 0.99;

        protected override bool StopTraining(int epoch, double loss) => epoch == 1000;

        protected override void Callback(int epoch, double loss) { }
    }

    public class Optimizer2 : Rmsprop
    {
        protected override double Alpha => 0.9;

        protected override double LearningRate(int epoch) => 0.01;

        protected override bool StopTraining(int epoch, double loss) => epoch == 1000;
    }
}