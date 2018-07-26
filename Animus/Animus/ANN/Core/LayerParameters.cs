using Animus.Tensor;
using System.Collections.Generic;

namespace Animus.ANN.Core
{
    public class LayerParameters : IDifferentiable<LayerParameters>
    {
        public LayerParameters(Matrix weights, Vector bias)
        {
            Weights = weights;
            Bias = bias;
        }

        public Matrix Weights { get; }

        public Vector Bias { get; }

        public LayerParameters Pow(Scalar n) => new LayerParameters(Weights.Pow(n), Bias.Pow(n));

        public LayerParameters Hadamard(LayerParameters p) => new LayerParameters(Weights.Hadamard(p.Weights), Bias.Hadamard(p.Bias));

        public static LayerParameters Zero => new LayerParameters(Matrix.Zero, Vector.Zero);

        public static LayerParameters Sqrt(LayerParameters w) => new LayerParameters(TensorFunction.Sqrt(w.Weights), TensorFunction.Sqrt(w.Bias));

        public static IEnumerable<LayerParameters> Zeroes(int n)
        {
            for (int i = 0; i < n; i++)
            {
                yield return new LayerParameters(Matrix.Zero, Vector.Zero);
            }
        }

        #region differential

        public LayerParameters Primal => new LayerParameters(Weights.Primal, Bias.Primal);

        public LayerParameters Tangent => new LayerParameters(Weights.Tangent, Bias.Tangent);

        public LayerParameters Adjoint => new LayerParameters(Weights.Adjoint, Bias.Adjoint);

        public LayerParameters Forward(LayerParameters tangent, uint lvl) => new LayerParameters(Weights.Forward(tangent.Weights, lvl), Bias.Forward(tangent.Bias, lvl));

        public LayerParameters Reverse(uint lvl) => new LayerParameters(Weights.Reverse(lvl), Bias.Reverse(lvl));

        #endregion differential

        #region lp - lp

        public static LayerParameters operator +(LayerParameters w1, LayerParameters w2) => new LayerParameters(w1.Weights + w2.Weights, w1.Bias + w1.Bias);

        public static LayerParameters operator -(LayerParameters w1, LayerParameters w2) => new LayerParameters(w1.Weights - w2.Weights, w1.Bias - w1.Bias);

        public static LayerParameters operator *(LayerParameters w1, LayerParameters w2) => new LayerParameters(w1.Weights.Hadamard(w2.Weights), w1.Bias.Hadamard(w2.Bias));

        public static LayerParameters operator /(LayerParameters w1, LayerParameters w2) => new LayerParameters(w1.Weights.Hadamard(1 / w2.Weights), w1.Bias.Hadamard(1 / w2.Bias));

        #endregion lp - lp

        #region s - lp

        public static LayerParameters operator +(Scalar s, LayerParameters w) => new LayerParameters(s + w.Weights, s + w.Bias);

        public static LayerParameters operator *(Scalar s, LayerParameters w) => new LayerParameters(s * w.Weights, s * w.Bias);

        #endregion s - lp
    }
}