using static DiffSharp.AD.Float64;

namespace Animus.Tensor
{
    public static partial class TensorFunction
    {
        public static Scalar Log(Scalar s) => new Scalar(D.Log(s.scalar));

        public static Scalar Exp(Scalar s) => new Scalar(D.Exp(s.scalar));

        public static Scalar Sin(Scalar s) => new Scalar(D.Sin(s.scalar));

        public static Scalar Cos(Scalar s) => new Scalar(D.Cos(s.scalar));

        public static Scalar Tan(Scalar s) => new Scalar(D.Tan(s.scalar));

        public static Scalar Sqrt(Scalar s) => new Scalar(D.Sqrt(s.scalar));

        public static Scalar Sinh(Scalar s) => new Scalar(D.Sinh(s.scalar));

        public static Scalar Cosh(Scalar s) => new Scalar(D.Cosh(s.scalar));

        public static Scalar Tanh(Scalar s) => new Scalar(D.Tanh(s.scalar));

        public static Scalar Asin(Scalar s) => new Scalar(D.Asin(s.scalar));

        public static Scalar Acos(Scalar s) => new Scalar(D.Acos(s.scalar));

        public static Scalar Atan(Scalar s) => new Scalar(D.Atan(s.scalar));

        public static Scalar Abs(Scalar s) => new Scalar(D.Abs(s.scalar));

        public static Scalar ReLU(Scalar s) => new Scalar(D.ReLU(s.scalar));

        public static Scalar Sigmoid(Scalar s) => new Scalar(D.Sigmoid(s.scalar));

        public static Scalar LogSumExp(Scalar s) => new Scalar(D.LogSumExp(s.scalar));

        public static Scalar Max(Scalar s1, Scalar s2) => new Scalar(D.Max(s1.scalar, s2.scalar));

        public static Scalar Min(Scalar s1, Scalar s2) => new Scalar(D.Min(s1.scalar, s2.scalar));
    }
}