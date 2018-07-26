namespace Animus.ANN.Optimizer.Reinforcement
{
    public class ActionScore<TAction>
    {
        public TAction Action { get; set; }

        public double Score { get; set; }
    }
}