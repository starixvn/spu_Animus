using Animus.ANN.Core;
using System;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.DataMining
{
    public interface IOptimizer
    {
        Task<LayerParameterList> TrainAsync<TInput, TOutput>(IDataRepository<TInput, TOutput> repository, Func<LayerParameterList, TInput, TOutput> objective, LayerParameterList initial);
    }
}