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

            foreach (var iris in data.GroupBy(ir => ir.Type).SelectMany(g => Rng.SelectNoRepeat(g, 1)))
            {
                var output = flow.Model(p, iris);

                // check the output
                Console.WriteLine($"{iris.Type}: {output}");
            }
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