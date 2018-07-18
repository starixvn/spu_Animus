namespace Animus.ANN.Core
{
    public interface IDataFlow<TInput, TOutput>
    {
        TOutput Train(LayerParameterList parameters, TInput input);

        TOutput Model(LayerParameterList parameters, TInput input);
    }
}