using Animus.ANN.Types.Modifiers;
using Animus.Tensor;

namespace Animus.ANN.Types.Layers
{
    public abstract class InputLayer<TInput>
    {
        public int Neurons => Mod != null && Mod.ModdedNeurons.HasValue ? Mod.ModdedNeurons.Value : RawNeurons;

        public abstract InputModifier Mod { get; }

        protected abstract int RawNeurons { get; }

        protected abstract Vector RawTransform(TInput data);

        public Vector Train(TInput data) => Mod != null ? Mod.Train(RawTransform(data)) : RawTransform(data);

        public Vector Model(TInput data) => Mod != null ? Mod.Model(RawTransform(data)) : RawTransform(data);
    }
}