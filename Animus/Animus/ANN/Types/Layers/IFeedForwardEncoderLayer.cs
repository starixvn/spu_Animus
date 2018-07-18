using Animus.ANN.Core;
using Animus.Tensor;

namespace Animus.ANN.Types.Layers
{
    public interface IFeedForwardEncoderLayer : IFeedForwardLayer
    {
        IDataFlow<Vector, Vector> BuildTransposeFlow(int offset);
    }
}