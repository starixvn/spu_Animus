using Animus.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Types.Modifiers
{
    public static class ModifierFactory
    {
        public static InputModifier ZeroMean<T>(IEnumerable<T> data, Func<T, Vector> mapper) => new ZeroMeanInputModifier(data.Select(mapper));

        public static InputModifier StandardCorrelation<T>(IEnumerable<T> data, Func<T, Vector> mapper, Func<Matrix, int> filter) => new StandardCorrelationInputModifier(data.Select(mapper), filter);

        public static IModifier BernoulliDropout(double p) => new BernoulliDropoutModifier(p);

        public static IModifier GaussianDropout(double std) => new GaussianDropout(std);
    }
}