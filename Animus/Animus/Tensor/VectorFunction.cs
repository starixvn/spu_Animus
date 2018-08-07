using static DiffSharp.AD.Float64;

namespace Animus.Tensor
{
    public static partial class TensorFunction
    {
        public static Vector Log(Vector v) => new Vector(DV.Log(v.vector));

        public static Vector Exp(Vector v) => new Vector(DV.Exp(v.vector));

        public static Vector Sin(Vector v) => new Vector(DV.Sin(v.vector));

        public static Vector Cos(Vector v) => new Vector(DV.Cos(v.vector));

        public static Vector Tan(Vector v) => new Vector(DV.Tan(v.vector));

        public static Vector Sqrt(Vector v) => new Vector(DV.Sqrt(v.vector));

        public static Vector Sinh(Vector v) => new Vector(DV.Sinh(v.vector));

        public static Vector Cosh(Vector v) => new Vector(DV.Cosh(v.vector));

        public static Vector Tanh(Vector v) => new Vector(DV.Tanh(v.vector));

        public static Vector Asin(Vector v) => new Vector(DV.Asin(v.vector));

        public static Vector Acos(Vector v) => new Vector(DV.Acos(v.vector));

        public static Vector Atan(Vector v) => new Vector(DV.Atan(v.vector));

        public static Vector Abs(Vector v) => new Vector(DV.Abs(v.vector));

        public static Vector ReLU(Vector v) => new Vector(DV.ReLU(v.vector));

        public static Vector Sigmoid(Vector v) => new Vector(DV.Sigmoid(v.vector));

        public static Vector Softmax(Vector v) => new Vector(DV.SoftMax(v.vector));

        public static Scalar L2Norm2(Vector v) => new Scalar(DV.L2NormSq(v.vector));
    }
}