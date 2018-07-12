using Animus.Tensor;
using System.Collections.Generic;

namespace Animus.ANN.Types.Modifiers
{
    internal class ZeroMeanInputModifier : InputModifier
    {
        private Vector mean;

        public ZeroMeanInputModifier(IEnumerable<Vector> data) => mean = data.Mean();

        public override Vector Model(Vector input) => input - mean;

        public override Vector Train(Vector input) => input - mean;
    }
}