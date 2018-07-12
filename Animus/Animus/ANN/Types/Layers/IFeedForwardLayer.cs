using Animus.ANN.Core;
using Animus.Tensor;

namespace Animus.ANN.Types.Layers
{
    public interface IFeedForwardLayer
    {
        int Neurons { get; }

        IDataFlow<Vector, Vector> BuildFlow(int offset);

        LayerParameters Initialize(int prevLayer);
    }
}