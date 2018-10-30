using Animus.ANN.Core;
using System;

namespace Animus.ANN.Optimizer
{
    public class GenericModel<TInput, TOutput> : IModel<TInput, TOutput>
    {
        private LayerParameterList parameters;
        private Func<LayerParameterList, TInput, TOutput> modelFunc;

        public GenericModel(LayerParameters[] parameters, Func<LayerParameterList, TInput, TOutput> modelFunc)
        {
            this.parameters = new LayerParameterList(parameters);

            this.modelFunc = modelFunc;
        }

        public LayerParameters[] Parameters => parameters.Raw;

        public TOutput Compute(TInput input) => modelFunc(parameters, input);
    }
}