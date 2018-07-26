using Animus.ANN.Core;

namespace Animus.ANN.Optimizer.Reinforcement
{
    public interface IAsyncOptimizer
    {
        LayerParameterList Update(int game, int turn, LayerParameters[] gradient, LayerParameterList currentParameters);
    }
}