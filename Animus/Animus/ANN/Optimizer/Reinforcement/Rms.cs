using Animus.ANN.Core;
using System.Linq;

namespace Animus.ANN.Optimizer.Reinforcement
{
    public abstract class Rms : IAsyncOptimizer
    {
        private static LayerParameters[] g = null;

        virtual protected double Alpha => 0.9;

        protected abstract double LearningRate(int game, int turn);

        public LayerParameterList Update(int game, int turn, LayerParameters[] gradient, LayerParameterList currentParameters)
        {
            var e = LearningRate(game, turn);

            if (g is null)
            {
                g = gradient.Select(gp => (1 - Alpha) * gp.Pow(2)).ToArray();
            }
            else
            {
                g = g.Select((gp, i) => Alpha * gp + (1 - Alpha) * gradient[i].Pow(2)).ToArray();
            }

            return new LayerParameterList(currentParameters.Select((p, i) => p - e * gradient[i] / LayerParameters.Sqrt(1e-7 + g[i])));
        }
    }
}