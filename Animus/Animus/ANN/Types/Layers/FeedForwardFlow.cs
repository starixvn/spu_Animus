using Animus.ANN.Core;
using Animus.ANN.Types.Modifiers;
using Animus.Tensor;
using System;

namespace Animus.ANN.Types.Layers
{
    internal class FeedForwardFlow : IDataFlow<Vector, Vector>
    {
        private readonly int offset;
        private readonly Func<Vector, Vector> activator;
        private readonly IModifier modifier;

        public FeedForwardFlow(int offset, Func<Vector, Vector> activator, IModifier modifier)
        {
            this.offset = offset;
            this.activator = activator;
            this.modifier = modifier;
        }

        public Vector Model(LayerParameters[] parameters, Vector input)
        {
            var sum = parameters[offset].Linear(input);

            var normalize = activator(sum);

            var modded = modifier.Model(normalize);

            return modded;
        }

        public Vector Train(LayerParameters[] parameters, Vector input)
        {
            var sum = parameters[offset].Linear(input);

            var normalize = activator(sum);

            var modded = modifier.Train(normalize);

            return modded;
        }
    }
}