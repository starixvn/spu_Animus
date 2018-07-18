using Animus.ANN.Core;
using Animus.Tensor;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.DataMining
{
    internal static class Helper
    {
        public static Task<Scalar> CalculateBatch<TInput, TOutput>(LayerParameterList parameters, IDataRepository<TInput, TOutput> repository, Func<LayerParameterList, TInput, TOutput> objective)
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