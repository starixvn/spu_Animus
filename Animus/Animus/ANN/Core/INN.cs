using System.Collections.Generic;

namespace Animus.ANN.Core
{
    public interface INN<TInput, TOutput>
    {
        int Layers { get; }

        IEnumerable<LayerParameters> Initialize();

        IDataFlow<TInput, TOutput> BuildFlow();
    }
}