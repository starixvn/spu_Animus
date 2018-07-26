using System.Collections.Generic;

namespace Animus.ANN.Optimizer.Reinforcement
{
    public interface IEnvironment<TState, TAction>
    {
        TState CreateInitialState();

        IEnumerable<TAction> GetActions(TState state);

        TState NextState(TState state, TAction action);

        bool IsTerminalState(TState state);
    }
}