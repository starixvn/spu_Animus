using Animus.ANN.Types.Modifiers;
using Animus.Tensor;
using System;

namespace Animus.ANN.Types.Layers
{
    internal class StandardInputLayer<T> : InputLayer<T>
    {
        private readonly Func<T, Vector> transformer;
        private readonly int neurons;

        public StandardInputLayer(Func<T, Vector> transformer, int neurons)
        {
            this.transformer = transformer;
            this.neurons = neurons;
        }

        public override InputModifier Mod => null;

        protected override int RawNeurons => neurons;

        protected override Vector RawTransform(T data) => transformer(data);
    }
}