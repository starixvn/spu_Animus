using Animus.ANN.Core;
using Animus.ANN.Types.Modifiers;
using Animus.Tensor;
using System;

namespace Animus.ANN.Types.Layers
{
    public abstract class FeedForwardLayer : IFeedForwardLayer
    {
        public abstract IModifier Modifier { get; }

        public abstract int Neurons { get; }

        public abstract Func<Vector, Vector> Activator { get; }

        public IDataFlow<Vector, Vector> BuildFlow(int offset)
        {
            if(Modifier == null)
            {
                throw new InvalidOperationException("Modifier can't be null, use standard layer for no modifier layer.");
            }

            return new FeedForwardFlow(offset, Activator, Modifier);
        }

        public abstract LayerParameters Initialize(int prevLayer);
    }
}