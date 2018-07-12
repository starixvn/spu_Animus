using Animus.ANN.Core;
using System;

namespace Animus.ANN
{
    internal class MappedInputModel<TInput, TInput1, TOutput> : IModel<TInput, TOutput>
    {
        private IModel<TInput1, TOutput> model;

        private Func<TInput, TInput1> mapper;

        public MappedInputModel(IModel<TInput1, TOutput> model, Func<TInput, TInput1> mapper)
        {
            Parameters = model.Parameters;
            this.model = model;
            this.mapper = mapper;
        }

        public LayerParameters[] Parameters { get; }

        public TOutput Compute(TInput input)
        {
            return model.Compute(mapper(input));
        }
    }
}