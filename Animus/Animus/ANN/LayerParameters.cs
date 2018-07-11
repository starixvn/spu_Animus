using Animus.Tensor;

namespace Animus.ANN
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

        #region differentiate

        public LayerParameters Primal => new LayerParameters(Weights.Primal, Bias.Primal);

        public LayerParameters Tangent => new LayerParameters(Weights.Tangent, Bias.Tangent);

        public LayerParameters Adjoint => new LayerParameters(Weights.Adjoint, Bias.Adjoint);

        public LayerParameters Forward(LayerParameters tangent, uint lvl)
        {
            throw new System.NotImplementedException();
        }

        public LayerParameters Reverse(uint lvl)
        {
            throw new System.NotImplementedException();
        }

        #endregion differentiate

        #region lp - lp

        public static LayerParameters operator +(LayerParameters w1, LayerParameters w2) => new LayerParameters(w1.Weights + w2.Weights, w1.Bias + w1.Bias);

        public static LayerParameters operator -(LayerParameters w1, LayerParameters w2) => new LayerParameters(w1.Weights - w2.Weights, w1.Bias - w1.Bias);

        #endregion lp - lp

        #region s - lp

        public static LayerParameters operator *(Scalar s, LayerParameters w) => new LayerParameters(s * w.Weights, s * w.Bias);

        #endregion s - lp
    }
}