namespace Animus.ANN.Core
{
    public interface IDataFlow<TInput, TOutput>
    {
        TOutput Train(LayerParameters[] parameters, TInput input);

        TOutput Model(LayerParameters[] parameters, TInput input);
    }
}