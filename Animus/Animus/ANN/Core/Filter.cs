using Animus.Tensor;

namespace Animus.ANN.Core
{
    public static class Filter
    {
        public static Vector Linear(this LayerParameters layer, Vector input) => layer.Weights * input + layer.Bias;
    }
}