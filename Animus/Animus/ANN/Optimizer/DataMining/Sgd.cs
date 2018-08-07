#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed

using Animus.ANN.Core;
using Animus.Tensor;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.DataMining
{
    public abstract class Sgd : IOptimizer
    {
        protected LayerParameterList CurrentParameters { get; private set; }

        protected LayerParameters[] AccumulatedDirection { get; private set; }

        protected abstract double LearningRate(int epoch);

        protected abstract double Momentum(int epoch);

        protected abstract bool StopTraining(int epoch, double loss);

        protected virtual void Callback(int epoch, double loss) => Console.WriteLine($"{epoch} - {loss}");

        public async Task<LayerParameterList> TrainAsync<TInput, TOutput>(IDataRepository<TInput, TOutput> repository, Func<LayerParameterList, TInput, TOutput> objective, LayerParameterList initial)
        {
            Func<LayerParameterList, Task<Scalar>> f = p => Helper.CalculateBatch(p, repository, objective);

            CurrentParameters = initial;
            AccumulatedDirection = LayerParameters.Zeroes(initial.Count).ToArray();

            var epoch = 0;
            var loss = double.PositiveInfinity;

            do
            {
                var learningRate = LearningRate(epoch);
                var momentum = Momentum(epoch);

                var (functionValue, gradient) = await f.Gradient(CurrentParameters);

                loss = (double)functionValue;

                //update step value
                AccumulatedDirection = AccumulatedDirection.Select((dir, idx) => momentum * dir - learningRate * gradient[idx]).ToArray();

                //update parameters according to step
                CurrentParameters = new LayerParameterList(CurrentParameters.Select((p, i) => p + AccumulatedDirection[i]));

                //replicate so that each callback is independent
                var callbackEpoch = ++epoch;
                var callbackLoss = loss;

                // no need to wait for call back
                Task.Factory.StartNew(() => Callback(callbackEpoch, callbackLoss));
            } while (!StopTraining(epoch, loss));

            return CurrentParameters;
        }
    }
}

#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed