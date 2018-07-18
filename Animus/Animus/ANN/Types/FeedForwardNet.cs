using Animus.ANN.Core;
using Animus.ANN.Types.Layers;
using Animus.Tensor;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Types
{
    public class FeedForwardNet<TInput> : INN<TInput, Vector>
    {
        public FeedForwardNet(InputLayer<TInput> inputLayer, IEnumerable<IFeedForwardLayer> hiddenLayers, IFeedForwardLayer outputLayer)
        {
            InputLayer = inputLayer;
            HiddenLayers = hiddenLayers;
            OutputLayer = outputLayer;
        }

        public InputLayer<TInput> InputLayer { get; }

        public IEnumerable<IFeedForwardLayer> HiddenLayers { get; }

        public IFeedForwardLayer OutputLayer { get; }

        public int Layers => HiddenLayers.Count() + 1;

        public IDataFlow<TInput, Vector> BuildFlow() => new FeedForwardNetFlow<TInput>(this);

        public IEnumerable<LayerParameters> Initialize()
        {
            var n = InputLayer.Neurons;

            foreach (var layer in HiddenLayers)
            {
                yield return layer.Initialize(n);

                n = layer.Neurons;
            }

            yield return OutputLayer.Initialize(n);
        }
    }
}