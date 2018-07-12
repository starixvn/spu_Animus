using Animus.ANN.Core;
using Animus.Tensor;
using System;

namespace Animus.ANN.Types.Layers
{
    internal class StandardForwardLayerFlow : IDataFlow<Vector, Vector>
    {
        private int offset;

        private Func<Vector, Vector> activator;

        public StandardForwardLayerFlow(int offset, Func<Vector, Vector> activator)
        {
            this.offset = offset;
            this.activator = activator;
        }

        public Vector Model(LayerParameters[] parameters, Vector input) => activator(parameters[offset].Linear(input));

        public Vector Train(LayerParameters[] parameters, Vector input) => activator(parameters[offset].Linear(input));
    }
}