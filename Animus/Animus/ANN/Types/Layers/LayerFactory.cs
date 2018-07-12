using Animus.ANN.Core;
using Animus.Tensor;
using System;

namespace Animus.ANN.Types.Layers
{
    public static class LayerFactory
    {
        public static InputLayer<TInput> MakeInputLayer<TInput>(Func<TInput, Vector> mapper, int neurons)
        {
            return new StandardInputLayer<TInput>(mapper, neurons);
        }

        public static IFeedForwardLayer MakeTanhLayer(int n)
        {
            return new StandardLayer(TensorFunction.Tanh, n, (prev, curr) => Initializer.XavierGaussian(1, prev, curr));
        }

        public static IFeedForwardLayer MakeSigmoidLayer(int n)
        {
            return new StandardLayer(TensorFunction.Sigmoid, n, (prev, curr) => Initializer.SigmoidGaussian(prev, curr));
        }
    }
}