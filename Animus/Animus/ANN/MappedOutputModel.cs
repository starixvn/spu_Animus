using Animus.ANN.Core;
using System;

namespace Animus.ANN
{
    public class MappedOutputModel<TInput, TOutput1, TOutput> : IModel<TInput, TOutput>
    {
        private IModel<TInput, TOutput1> model;

        private Func<TOutput1, TOutput> mapper;

        public MappedOutputModel(IModel<TInput, TOutput1> model, Func<TOutput1, TOutput> mapper)
        {
            this.model = model;
            this.mapper = mapper;

            Parameters = model.Parameters;
        }

        public LayerParameters[] Parameters { get; }

        public TOutput Compute(TInput input)
        {
            return mapper(model.Compute(input));
        }
    }
}