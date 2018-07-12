using Animus.Tensor;

namespace Animus.ANN.Types.Modifiers
{
    public abstract class InputModifier
    {
        public virtual int? ModdedNeurons { get; }

        public abstract Vector Model(Vector input);

        public abstract Vector Train(Vector input);
    }
}