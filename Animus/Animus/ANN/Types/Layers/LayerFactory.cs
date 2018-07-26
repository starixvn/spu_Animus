using Animus.ANN.Core;
using Animus.Tensor;
using System;

namespace Animus.ANN.Types.Layers
{
    public static class LayerFactory
    {
        public static InputLayer<TInput> MakeInputLayer<TInput>(Func<TInput, Vector> mapper, int neurons) => new StandardInputLayer<TInput>(mapper, neurons);

        public static IFeedForwardLayer MakeLinearLayer(int n) => new StandardLayer(x => x, n, (prev, curr) => Initializer.XavierUniform(1, prev, curr));

        public static IFeedForwardLayer MakeTanhLayer(int n) => new StandardLayer(TensorFunction.Tanh, n, (prev, curr) => Initializer.XavierGaussian(1, prev, curr));

        public static IFeedForwardLayer MakeSigmoidLayer(int n) => new StandardLayer(TensorFunction.Sigmoid, n, Initializer.SigmoidGaussian);

        public static IFeedForwardLayer MakeSoftmaxLayer(int n) => new StandardLayer(TensorFunction.Softmax, n, (prev, curr) => Initializer.XavierUniform(1, prev, curr));

        public static IFeedForwardLayer MakeReLuLayer(int n) => new StandardLayer(TensorFunction.ReLU, n, Initializer.ReLuGaussian);
    }
}