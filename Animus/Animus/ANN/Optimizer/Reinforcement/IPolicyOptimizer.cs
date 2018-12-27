using System.Collections.Generic;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.Reinforcement
{
    public interface IPolicyOptimizer<TState, TAction, TAgent>
    {
        double Discount { get; }

        double Reward(TState state, TAction action, TState nextState);

        int ActionIndex(TAction action);

        int Threads { get; }

        void ThreadCallback(int thread, int game, double point);

        IModel<TState, IEnumerable<ActionScore<TAction>>> Train(IEnvironment<TState, TAction> environment, TAgent agent);

        bool Terminate(int thread, int game);
    }
}