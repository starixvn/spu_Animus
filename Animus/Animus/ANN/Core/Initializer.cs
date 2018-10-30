using Animus.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Core
{
    public class Initializer
    {
        private static LayerParameters FillParameters(IEnumerable<double> data, int prevLayer, int currLayer)
        {
            var idx = 0;
            var weights = new double[currLayer, prevLayer];

            foreach (var item in data.Take(currLayer * prevLayer))
            {
                weights[idx / prevLayer, idx % prevLayer] = item;

                idx++;
            }

            return new LayerParameters(new Matrix(weights), Vector.Zero);
        }

        public static LayerParameters XavierUniform(double df0, int prevLayer, int currLayer)
        {
            var range = Math.Sqrt(6.0 / (prevLayer + currLayer) / df0);
            var rng = Rng.Uniform(-range, range);

            return FillParameters(rng.Samples(), prevLayer, currLayer);
        }

        public static LayerParameters XavierGaussian(double df0, int prevLayer, int currLayer)
        {
            var std = Math.Sqrt(2.0 / (prevLayer + currLayer) / df0);
            var rng = Rng.Normal(0, std);

            return FillParameters(rng.Samples(), prevLayer, currLayer);
        }

        public static LayerParameters SigmoidGaussian(int prevLayer, int currLayer)
        {
            var rng = Rng.Normal(0, 3.6 / Math.Sqrt(currLayer));

            return FillParameters(rng.Samples(), prevLayer, currLayer);
        }

        public static LayerParameters ReLuGaussian(int prevLayer, int currLayer)
        {
            var rng = Rng.Normal(0, Math.Sqrt(2.0 / currLayer));

            return FillParameters(rng.Samples(), prevLayer, currLayer);
        }
    }
}