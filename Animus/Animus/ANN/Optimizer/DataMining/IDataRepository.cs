using Animus.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Animus.ANN.Optimizer.DataMining
{
    public interface IDataRepository<TInput, TOutput>
    {
        IEnumerable<TInput> Sample();

        Scalar Loss(TInput sample, TOutput);
    }
}
