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
            // create input node in forward mode
            var forward = x.Forward(Scalar.One, GlobalTagger.Next);

            // calculate + release graph as it moves forward
            var result = f(forward);

            return (result.Primal, result.Tagent.Primal);
        }

        public static (Scalar functionValue, T gradient) Gradient<T>(this Func<T, Scalar> f, T x) where T : IDifferentiable<T>
        {
            // create input nodes
            var reverse = x.Reverse(GlobalTagger.Next);

            // build full graph using f
            var result = f(reverse);

            //go from output to input nodes 
            DOps.reverseProp(D.One, result.scalar);

            return (result.Primal, reverse.Adjoint);
        }
    }
}