using Animus.ANN.Core;
using Animus.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.Reinforcement
{
    /// <summary>
    /// Asynchronous Advantage Actor-Critic algorithm. This class isn't thread safe
    /// </summary>
    /// <typeparam name="TState"></typeparam>
    /// <typeparam name="TAction"></typeparam>
    public abstract class A3C<TState, TAction> : IPolicyOptimizer<TState, TAction, INN<TState, Vector>, IModel<TState, IEnumerable<(TAction action, double score)>>>
    {
        protected virtual double Beta => 0.01;

        public abstract double Discount { get; }

        public abstract int Threads { get; }

        public abstract int ActionIndex(TAction action);

        public abstract double Reward(TState state, TAction action, TState nextState);

        public abstract bool Terminate(int thread, int game);

        public abstract IAsyncOptimizer Optimizer { get; }

        public abstract bool StopAndUpdate(int game, int turn);

        public virtual void ThreadCallback(int thread, int game, double point)
        {
            Console.WriteLine($"{thread} - {game} - {point}");
        }

        private LayerParameterList parameters;

        private IEnvironment<TState, TAction> environment;

        private IDataFlow<TState, Vector> calculator;

        private IAsyncOptimizer optimizer;

        public async Task<IModel<TState, IEnumerable<(TAction action, double score)>>> Train(IEnvironment<TState, TAction> environment, INN<TState, Vector> agent)
        {
            //intialize thread share objects
            parameters = new LayerParameterList(agent.Initialize());
            this.environment = environment;
            calculator = agent.BuildFlow();
            optimizer = Optimizer;

            //create thread index
            var threads = Enumerable.Range(0, Threads).ToList();

            // no limit on parallelism
            var option = new ParallelOptions
            {
                MaxDegreeOfParallelism = -1,
            };

            //run each task in parallel and wait
            await Task.Factory.StartNew(() => Parallel.For(0, Threads, option, ExecuteThread));

            return new A3CModel<TState, TAction>(parameters, calculator, environment, this);
        }

        private void ExecuteThread(int thread)
        {
            var state = environment.CreateInitialState();

            var game = 1;
            var turn = 1;
            var reward = .0;

            while (!Terminate(thread, game))
            {
                var threadParameter = parameters;

                var (finalState, records) = Play(threadParameter, game, turn, state);

                var newReward = ProcessData(threadParameter, finalState, records);

                Learn(game, turn, threadParameter, records);

                state = finalState;
                game++;
                turn = turn + records.Count - 1;
                reward += newReward;

                if (environment.IsTerminalState(finalState))
                {
                    // replicate so that later loop won't modify it
                    var replicateGame = game - 1;
                    var replicateReward = reward;

                    new Thread(() => ThreadCallback(thread, replicateGame, replicateReward)).Start();

                    state = environment.CreateInitialState();

                    turn = 1;
                    reward = 0;
                }
            }
        }

        private (Scalar value, Vector policy) Calculate(LayerParameterList parameters, TState state)
        {
            var valuePolicy = calculator.Train(parameters, state);

            var actions = environment.GetActions(state);

            var policy = environment.GetActions(state)
                .Select(a => valuePolicy[ActionIndex(a) + 1]);

            var normalizedPolicy = TensorFunction.Softmax(Vector.OfArray(policy.ToArray()));

            return (valuePolicy[0], normalizedPolicy);
        }

        private (TState finalState, Stack<Record<TState, TAction>> records) Play(LayerParameterList threadParameters, int game, int turn, TState state)
        {
            var result = new Stack<Record<TState, TAction>>();

            var currentState = state;

            while (!environment.IsTerminalState(state) && !StopAndUpdate(game, turn))
            {
                var (value, policy) = Calculate(parameters, currentState);

                var rng = Rng.Category((double[])policy);

                var index = rng.Sample();

                var action = environment.GetActions(currentState).ElementAt(rng.Sample());

                var nextState = environment.NextState(currentState, action);

                result.Push(new Record<TState, TAction>
                {
                    Action = action,
                    Index = index,
                    CurrentState = currentState,
                    NextState = nextState
                });

                currentState = nextState;
            }

            return (currentState, result);
        }

        private double ProcessData(LayerParameterList threadParameters, TState finalState, Stack<Record<TState, TAction>> records)
        {
            var value = .0;

            if (!environment.IsTerminalState(finalState))
            {
                var g = Calculate(threadParameters, finalState);

                value = (double)g.value;
            }

            var totalReward = .0;

            foreach (var record in records)
            {
                var reward = Reward(record.CurrentState, record.Action, record.NextState);

                value = reward + Discount * value;

                record.StateValue = value;

                totalReward += reward;
            }

            return totalReward;
        }

        private void Learn(int game, int turn, LayerParameterList threadParameters, IEnumerable<Record<TState, TAction>> data)
        {
            Func<LayerParameterList, Scalar> f = p => CalculateLoss(p, data);

            var (functionValue, gradient) = f.Gradient(threadParameters);

            parameters = optimizer.Update(game, turn, gradient.Raw, parameters);
        }

        private Scalar CalculateLoss(LayerParameterList threadParameters, IEnumerable<Record<TState, TAction>> data)
        {
            return data.Aggregate(Scalar.Zero, (acc, r) => acc + CalculateLoss(threadParameters, r));
        }

        private Scalar CalculateLoss(LayerParameterList threadParameters, Record<TState, TAction> record)
        {
            var (value, policy) = Calculate(threadParameters, record.CurrentState);

            var policyLoss = -(record.StateValue - (double)value) * TensorFunction.Log(policy[record.Index]);

            var valueLoss = (record.StateValue - value).Pow(2);

            var entropy = -policy * TensorFunction.Log(policy);

            return policyLoss + valueLoss - Beta * entropy;
        }
    }
}