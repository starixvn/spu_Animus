using Animus.ANN.Core;
using System;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.DataMining
{
    public interface IOptimizer
    {
        Task<LayerParameterList> Train<TInput, TOuput>(IDataRepository<TInput, TOuput> repository, Func<LayerParameterList, TInput, TOuput> objective, LayerParameterList initial);
    }
}