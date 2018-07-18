using Animus.ANN.Core;
using Animus.ANN.Types.Layers;
using Animus.Tensor;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Types
{
    internal class FeedForwardNetFlow<TInput> : IDataFlow<TInput, Vector>
    {
        private readonly InputLayer<TInput> inputLayer;

        private readonly IEnumerable<IDataFlow<Vector, Vector>> hiddenLayers;

        private readonly IDataFlow<Vector, Vector> outputLayer;

        public FeedForwardNetFlow(FeedForwardNet<TInput> network)
        {
            inputLayer = network.InputLayer;

            hiddenLayers = network.HiddenLayers.Select((layer, idx) => layer.BuildFlow(idx)).ToList();

            outputLayer = network.OutputLayer.BuildFlow(network.Layers - 1);
        }

        public Vector Model(LayerParameters[] parameters, TInput input)
        {
            var result = inputLayer.Model(input);

            foreach (var layer in hiddenLayers)
            {
                result = layer.Model(parameters, result);
            }

            return outputLayer.Model(parameters, result);
        }

        public Vector Train(LayerParameters[] parameters, TInput input)
        {
            var result = inputLayer.Train(input);

            foreach (var layer in hiddenLayers)
            {
                result = layer.Train(parameters, result);
            }

            return outputLayer.Train(parameters, result);
        }
    }
}