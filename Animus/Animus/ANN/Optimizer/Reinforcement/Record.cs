namespace Animus.ANN.Optimizer.Reinforcement
{
    public class Record<TState, TAction>
    {
        public TState CurrentState { get; set; }

        public TAction Action { get; set; }

        public int Index { get; set; }

        public TState NextState { get; set; }

        public double StateValue { get; set; }
    }
}