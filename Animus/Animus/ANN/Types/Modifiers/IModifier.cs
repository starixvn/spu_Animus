using Animus.Tensor;

namespace Animus.ANN.Types.Modifiers
{
    public interface IModifier
    {
        Vector Train(Vector input);

        Vector Model(Vector input);
    }
}