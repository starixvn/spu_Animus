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
    public abstract class A3C<TState, TAction> : IPolicyOptimizer<TState, TAction, INN<TState, Vector>>
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

        public async Task<IModel<TState, IEnumerable<ActionScore<TAction>>>> Train(IEnvironment<TState, TAction> environment, INN<TState, Vector> agent)
        {
            //intialize thread share objects
            parameters = new LayerParameterList(agent.Initialize());
            this.environment = environment;
            calculator = agent.BuildFlow();
            optimizer = Optimizer;

            var q = new Queue<Thread>(Threads);

            for (int i = 0; i < Threads; i++)
            {
                var local = i;
                var thread = new Thread(a => ExecuteThread((int)a));
                thread.Start(i);
                q.Enqueue(thread);
            }

            await Task.Factory.StartNew(() =>
            {
                foreach (var item in q)
                {
                    item.Join();
                }
            });

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
                //sync thread to global
                var threadParameter = parameters;

                //play the game
                var (finalState, records, finalTurn) = Play(threadParameter, game, turn, state);

                //calculate reward
                var newReward = ProcessData(threadParameter, finalState, records);

                //update global
                Learn(game, turn, threadParameter, records);

                state = finalState;
                game++;
                turn = finalTurn;
                reward += newReward;

                if (environment.IsTerminalState(finalState))
                {
                    ThreadCallback(thread, game - 1, reward);

                    state = environment.CreateInitialState();

                    turn = 1;
                    reward = 0;
                }
            }
        }

        private (Scalar value, Vector policy) Calculate(LayerParameterList parameters, TState state)
        {
            var valuePolicy = calculator.Train(parameters, state);

            var policy = environment.GetActions(state)
                .Select(a => valuePolicy[ActionIndex(a) + 1]);

            var normalizedPolicy = TensorFunction.Softmax(Vector.OfSequence(policy.ToArray()));

            return (valuePolicy[0], normalizedPolicy);
        }

        private (TState finalState, Stack<Record<TState, TAction>> records, int finalTurn) Play(LayerParameterList threadParameters, int game, int turn, TState state)
        {
            var result = new Stack<Record<TState, TAction>>();

            var currentState = state;

            var finalTurn = turn;

            while (!environment.IsTerminalState(currentState) && !StopAndUpdate(game, turn))
            {
                // get action distribution in policy
                var (value, policy) = Calculate(threadParameters, currentState);

                // create a category distribution based on policy
                var rng = Rng.Category((double[])policy);

                // get a random action index
                var index = rng.Sample();

                // pick an action based on the index
                var action = environment.GetActions(currentState).ElementAt(index);

                // play the game to get next state
                var nextState = environment.NextState(currentState, action);

                // save data in stack
                result.Push(new Record<TState, TAction>
                {
                    Action = action,
                    Index = index,
                    CurrentState = currentState,
                    NextState = nextState
                });

                currentState = nextState;

                finalTurn++;
            }

            return (currentState, result, finalTurn);
        }

        private double ProcessData(LayerParameterList threadParameters, TState finalState, Stack<Record<TState, TAction>> records)
        {
            // finalState value
            var value = .0;

            // check if finalState is not game over
            if (!environment.IsTerminalState(finalState))
            {
                var g = Calculate(threadParameters, finalState);

                value = (double)g.value;
            }

            // total reward after playing the game
            var totalReward = .0;

            foreach (var record in records)
            {
                // get reward for this record
                var reward = Reward(record.CurrentState, record.Action, record.NextState);

                // calculate new value according to q-learning wiki
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

            return policyLoss + 0.5 * valueLoss - Beta * entropy;
        }
    }
}