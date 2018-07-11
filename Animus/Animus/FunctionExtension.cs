using Animus.Tensor;
using System;
using static DiffSharp.AD.Float64;
using static DiffSharp.Util;

namespace Animus
{
    public static class FunctionExtension
    {
        public static (Scalar functionValue, Scalar derivative) Derivative(this Func<Scalar, Scalar> f, Scalar x)
        {
            var forward = x.Forward(Scalar.One, GlobalTagger.Next);

            var result = f(forward);

            return (result.Primal, result.Tagent.Primal);
        }

        public static (Scalar functionValue, T gradient) Gradient<T>(this Func<T, Scalar> f, T x) where T : IDifferentiable<T>
        {
            var reverse = x.Reverse(GlobalTagger.Next);

            var result = f(reverse);

            DOps.reverseProp(D.One, result.scalar);

            return (result.Primal, reverse.Adjoint);
        }
    }
}