using System;

namespace Animus.ANN
{
    public static class ModelExtension
    {
        public static IModel<TInput, TOutput> MapInput<TInput, TInput1, TOutput>(this IModel<TInput1, TOutput> model, Func<TInput, TInput1> mapper)
        {
            return new MappedInputModel<TInput, TInput1, TOutput>(model, mapper);
        }

        public static IModel<TInput, TOutput> MapOutput<TInput, TOutput1, TOutput>(this IModel<TInput, TOutput1> model, Func<TOutput1, TOutput> mapper)
        {
            return new MappedOutputModel<TInput, TOutput1, TOutput>(model, mapper);
        }
    }
}