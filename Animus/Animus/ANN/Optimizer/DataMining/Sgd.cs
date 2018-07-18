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

        public async Task<LayerParameterList> Train<TInput, TOutput>(IDataRepository<TInput, TOutput> repository, Func<LayerParameterList, TInput, TOutput> objective, LayerParameterList initial)
        {
            Func<LayerParameterList, Task<Scalar>> f = p => CalculateBatch(p, repository, objective);

            CurrentParameters = initial;
            AccumulatedDirection = LayerParameters.Zeroes(initial.Count).ToArray();

            var epoch = 0;

            do
            {
                var learningRate = LearningRate(epoch);
                var momentum = Momentum(epoch);

                var (functionValue, gradient) = await f.Gradient(CurrentParameters);
                //update step value
                AccumulatedDirection = AccumulatedDirection.Select((dir, idx) => momentum * dir - learningRate * gradient[idx]).ToArray();

                //update parameters according to step
                CurrentParameters = new LayerParameterList(CurrentParameters.Select((p, i) => p + AccumulatedDirection[i]));

                // no need to wait for call back
                Task.Factory.StartNew((e) => Callback((int)e, functionValue), ++epoch);
            } while (!StopTraining(epoch));

            return CurrentParameters;
        }

        protected abstract double LearningRate(int epoch);

        protected abstract double Momentum(int epoch);

        protected abstract bool StopTraining(int epoch);

        protected virtual void Callback(int epoch, Scalar loss)
        {
        }

        private Task<Scalar> CalculateBatch<TInput, TOutput>(LayerParameterList parameters, IDataRepository<TInput, TOutput> repository, Func<LayerParameterList, TInput, TOutput> objective)
        {
            var data = repository.Sample().ToArray();

            return Task.Factory.StartNew(() => data
                    .AsParallel()
                    .AsUnordered()
                    .Select(d => repository.Loss(d, objective(parameters, d)))
                    .Aggregate(Scalar.Zero, (subtotal, loss) => subtotal + loss, (total, subtotal) => total + subtotal, total => total / data.Length));
        }
    }
}

#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed