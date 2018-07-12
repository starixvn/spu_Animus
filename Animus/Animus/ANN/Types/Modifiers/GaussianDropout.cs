using Animus.Tensor;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace Animus.ANN.Types.Modifiers
{
    internal class GaussianDropout : IModifier
    {
        private Normal rng;

        public GaussianDropout(double std)
        {
            rng = Rng.Normal(1, std);
        }

        public Vector Model(Vector input) => input;

        public Vector Train(Vector input)
        {
            var mod = new Vector(rng.Samples().Take(input.Length).ToArray());

            return input.Hadamard(mod);
        }
    }
}