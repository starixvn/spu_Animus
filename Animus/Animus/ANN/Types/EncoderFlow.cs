using Animus.ANN.Core;
using Animus.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Types
{
    internal class EncoderFlow : IDataFlow<Vector, Vector>
    {
        private readonly IList<IDataFlow<Vector, Vector>> encoders;

        private readonly IDataFlow<Vector, Vector> normalizer;

        private readonly IList<IDataFlow<Vector, Vector>> decoders;

        private readonly IDataFlow<Vector, Vector> rebuilder;

        public EncoderFlow(AutoEncoder encoder)
        {
            encoders = encoder.Encoders.Select((layer, idx) => layer.BuildFlow(idx)).ToList();

            normalizer = encoder.Normalizer.BuildFlow(encoder.Encoders.Count());

            decoders = encoder.Encoders.Reverse().Select((layer, idx) => layer.BuildFlow(idx)).ToList();

            rebuilder = encoder.Decoder.BuildFlow(encoder.Encoders.Count());
        }

        public Vector Model(LayerParameters[] parameters, Vector input)
        {
            var encoded = encoders.Aggregate(input, (acc, layer) => layer.Model(parameters, acc));

            return normalizer.Model(parameters, encoded);
        }

        public Vector Train(LayerParameters[] parameters, Vector input)
        {
            var encoded = encoders.Aggregate(input, (acc, layer) => layer.Train(parameters, acc));

            var normalized = normalizer.Train(parameters, encoded);

            var tightsWeight = CreateTightWeightsDecoded(parameters);

            //var decoded = decoders.Aggregate(normalized, (acc, layer) => layer.)

            throw new NotImplementedException();
        }

        private LayerParameters[] CreateTightWeightsDecoded(LayerParameters[] fullParameters)
        {
            var q = new Queue<LayerParameters>();

            for (int i = 0; i < decoders.Count + 1; i++)
            {
                q.Enqueue(new LayerParameters(fullParameters[encoders.Count - i].Weights, fullParameters[i + encoders.Count + 1].Bias));
            }

            return q.ToArray();
        }
    }
}