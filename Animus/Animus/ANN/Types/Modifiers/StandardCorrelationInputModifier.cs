using Animus.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Types.Modifiers
{
    internal class StandardCorrelationInputModifier : InputModifier
    {
        private Matrix transformingMatrix;

        public StandardCorrelationInputModifier(IEnumerable<Vector> data, Func<Matrix, int> Filter)
        {
            //get standard variation
            var std = TensorFunction.Sqrt(data.Variance());

            //normalize standard variation to 1, then calculate correlation
            var m = data.Select(v => v.Hadamard(1 / std)).Covariance();

            //get eigen matrix and values of correlation matrix
            var evd = m.Eigen();

            //pick out relevant dimension
            var n = Filter(evd.eigenValues);

            //transform normal coordinate to non correlated coordinate
            transformingMatrix = (evd.eigenVectors * evd.eigenValues).Inverse.Slice(0, n - 1, null, null);
        }

        public override int? ModdedNeurons => transformingMatrix.Rows;

        public override Vector Model(Vector input) => transformingMatrix * input;

        public override Vector Train(Vector input) => transformingMatrix * input;
    }
}