using Animus.Tensor;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace Animus.ANN.Types.Modifiers
{
    internal class BernoulliDropoutModifier : IModifier
    {
        private Bernoulli rng;

        public BernoulliDropoutModifier(double p)
        {
            rng = Rng.Bernoulli(p);
        }

        public Vector Model(Vector input)
        {
            return input * rng.P;
        }

        public Vector Train(Vector input)
        {
            var mod = new Vector(rng.Samples().Take(input.Length).Select(a => (double)a).ToArray());

            return input.Hadamard(mod);
        }
    }
}