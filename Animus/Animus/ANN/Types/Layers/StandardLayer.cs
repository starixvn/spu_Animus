using Animus.ANN.Core;
using Animus.Tensor;
using System;

namespace Animus.ANN.Types.Layers
{
    public class StandardLayer : IFeedForwardLayer
    {
        public StandardLayer(Func<Vector, Vector> activator, int neurons, Func<int, int, LayerParameters> initializer)
        {
            Activator = activator;
            Neurons = neurons;
            Initializer = initializer;
        }

        public Func<Vector, Vector> Activator { get; }

        public int Neurons { get; }

        public Func<int, int, LayerParameters> Initializer { get; }

        public IDataFlow<Vector, Vector> BuildFlow(int offset) => new StandardForwardLayerFlow(offset, Activator);

        public LayerParameters Initialize(int prevLayer) => Initializer(prevLayer, Neurons);
    }
}