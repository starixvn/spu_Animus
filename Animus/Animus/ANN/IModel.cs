using Animus.ANN.Core;

namespace Animus.ANN
{
    public interface IModel<TInput, TOutput>
    {
        LayerParameters[] Parameters { get; }

        TOutput Compute(TInput input);
    }
}