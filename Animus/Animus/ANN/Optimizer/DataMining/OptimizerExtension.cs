using Animus.ANN.Core;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.DataMining
{
    public static class OptimizerExtension
    {
        public async static Task<IModel<TInput, TOutput>> Train<TInput, TOutput>(this IOptimizer optimizer, INN<TInput, TOutput> network, IDataRepository<TInput, TOutput> repository)
        {
            var flow = network.BuildFlow();

            var parameters = await optimizer.TrainAsync(repository, flow.Train, new LayerParameterList(network.Initialize()));

            return new GenericModel<TInput, TOutput>(parameters.Raw, flow.Model);
        }
    }
}