using Animus.ANN.Core;
using Animus.ANN.Types.Layers;
using Animus.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Types
{
    public class AutoEncoder : INN<Vector, Vector>
    {
        public AutoEncoder(IEnumerable<IFeedForwardLayer> encoders, IFeedForwardLayer normalizer, IFeedForwardLayer decoder)
        {
            Encoders = encoders;
            Normalizer = normalizer;
            Decoder = decoder;
        }

        public int Layers => 2 * Encoders.Count() + 2;

        public IEnumerable<IFeedForwardLayer> Encoders { get; }

        public IFeedForwardLayer Normalizer { get; }

        public IFeedForwardLayer Decoder { get; }

        public IDataFlow<Vector, Vector> BuildFlow()
        {
            throw new NotImplementedException();
        }

        public IEnumerable<LayerParameters> Initialize()
        {
            var n = Decoder.Neurons;

            foreach (var layer in Encoders)
            {
                yield return layer.Initialize(n);
                n = layer.Neurons;
            }

            yield return Normalizer.Initialize(n);
            n = Normalizer.Neurons;

            foreach (var layer in Encoders.Reverse())
            {
                yield return LayerParameters.Zero;
                n = layer.Neurons;
            }

            yield return Decoder.Initialize(n);
        }

        public Func<Vector, Vector> BuildDecoder(LayerParameters[] parameters)
        {
            return null;
        }
    }
}