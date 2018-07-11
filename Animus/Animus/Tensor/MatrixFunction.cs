using static DiffSharp.AD.Float64;

namespace Animus.Tensor
{
    public static partial class TensorFunction
    {
        public static Matrix Log(Matrix m) => new Matrix(DM.Log(m.matrix));

        public static Matrix Exp(Matrix m) => new Matrix(DM.Exp(m.matrix));

        public static Matrix Sin(Matrix m) => new Matrix(DM.Sin(m.matrix));

        public static Matrix Cos(Matrix m) => new Matrix(DM.Cos(m.matrix));

        public static Matrix Tan(Matrix m) => new Matrix(DM.Tan(m.matrix));

        public static Matrix Sqrt(Matrix m) => new Matrix(DM.Sqrt(m.matrix));

        public static Matrix Sinh(Matrix m) => new Matrix(DM.Sinh(m.matrix));

        public static Matrix Cosh(Matrix m) => new Matrix(DM.Cosh(m.matrix));

        public static Matrix Tanh(Matrix m) => new Matrix(DM.Tanh(m.matrix));

        public static Matrix Asin(Matrix m) => new Matrix(DM.Asin(m.matrix));

        public static Matrix Acos(Matrix m) => new Matrix(DM.Acos(m.matrix));

        public static Matrix Atan(Matrix m) => new Matrix(DM.Atan(m.matrix));

        public static Matrix Abs(Matrix m) => new Matrix(DM.Abs(m.matrix));

        public static Matrix ReLU(Matrix m) => new Matrix(DM.ReLU(m.matrix));

        public static Matrix Sigmoid(Matrix m) => new Matrix(DM.Sigmoid(m.matrix));
    }
}