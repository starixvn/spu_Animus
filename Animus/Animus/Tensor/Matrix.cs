using MathNet.Numerics.LinearAlgebra;
using Microsoft.FSharp.Core;
using static DiffSharp.AD.Float64;

namespace Animus.Tensor
{
    public class Matrix : IDifferentiable<Matrix>
    {
        internal DM matrix;

        public Matrix(double[,] m) => matrix = DM.NewDM(m);

        internal Matrix(DM m) => matrix = m;

        public Scalar this[int i, int j] => new Scalar(matrix[i, j]);

        public int Rows => matrix.Rows;

        public int Columns => matrix.Cols;

        public int Length => Rows * Columns;

        public Scalar Determinant => new Scalar(DM.Det(matrix));

        public Scalar Trace => new Scalar(DM.Trace(matrix));

        public Vector Diagonal => new Vector(DM.Diagonal(matrix));

        public Matrix Transpose => new Matrix(DM.Transpose(matrix));

        public Matrix Inverse => new Matrix(DM.Inverse(matrix));

        public Vector Slice(int i, int? j1, int? j2)
        {
            var s = j1.HasValue ? j1.Value : 0;
            var e = j2.HasValue ? j2.Value : Columns - 1;

            return new Vector(matrix.GetSlice(i, new FSharpOption<int>(s), new FSharpOption<int>(e)));
        }

        public Vector Slice(int? i1, int? i2, int j)
        {
            var s = i1.HasValue ? i1.Value : 0;
            var e = i2.HasValue ? i2.Value : Rows - 1;

            return new Vector(matrix.GetSlice(new FSharpOption<int>(s), new FSharpOption<int>(e), j));
        }

        public Matrix Slice(int? i1, int? i2, int? j1, int? j2)
        {
            var s1 = i1.HasValue ? i1.Value : 0;
            var e1 = i2.HasValue ? i2.Value : Rows - 1;

            var s2 = j1.HasValue ? j1.Value : 0;
            var e2 = j2.HasValue ? j2.Value : Columns - 1;

            return new Matrix(matrix.GetSlice(
                    new FSharpOption<int>(s1),
                    new FSharpOption<int>(e1),
                    new FSharpOption<int>(s2),
                    new FSharpOption<int>(e2)
                ));
        }

        public (Matrix eigenValues, Matrix eigenVectors) Eigen()
        {
            var m = CreateMatrix.DenseOfArray((double[,])matrix);

            var evd = m.Evd();

            return (new Matrix(evd.D.ToArray()), new Matrix(evd.EigenVectors.ToArray()));
        }

        public override string ToString() => matrix.ToString().Replace("DM : ", "");

        public static Matrix Zero { get; } = new Matrix(DM.Zero);

        #region instance operator

        public Matrix Pow(Scalar s) => new Matrix(DM.Pow(matrix, s.scalar));

        public Matrix Pow(Matrix m) => new Matrix(DM.Pow(matrix, m.matrix));

        public Matrix Hadamard(Matrix m) => new Matrix(DM.op_DotMultiply(matrix, m.matrix));

        #endregion instance operator

        #region differential

        public Matrix Primal => new Matrix(matrix.P);

        public Matrix Tangent => new Matrix(matrix.T);

        public Matrix Adjoint => new Matrix(matrix.A);

        public Matrix Forward(Matrix tangent, uint lvl)
        {
            var result = matrix.GetForward(tangent.matrix, lvl);

            return new Matrix(result);
        }

        public Matrix Reverse(uint lvl)
        {
            var result = matrix.GetReverse(lvl);

            return new Matrix(result);
        }

        #endregion differential

        #region casting

        public static implicit operator Matrix(double[,] m) => new Matrix(m);

        public static explicit operator double[,] (Matrix m) => (double[,])m.matrix;

        #endregion casting

        #region M - M

        public static Matrix operator -(Matrix m1) => new Matrix(-m1.matrix);

        public static Matrix operator +(Matrix m1, Matrix m2) => new Matrix(m1.matrix + m2.matrix);

        public static Matrix operator -(Matrix m1, Matrix m2) => new Matrix(m1.matrix - m2.matrix);

        public static Matrix operator *(Matrix m1, Matrix m2) => new Matrix(m1.matrix * m2.matrix);

        #endregion M - M

        #region M - V, V - M

        public static Vector operator *(Matrix m, Vector v) => new Vector(m.matrix * v.vector);

        public static Vector operator *(Vector v, Matrix m) => new Vector(v.vector * m.matrix);

        #endregion M - V, V - M

        #region S - M

        public static Matrix operator +(Scalar s, Matrix m) => new Matrix(s.scalar + m.matrix);

        public static Matrix operator -(Scalar s, Matrix m) => new Matrix(s.scalar - m.matrix);

        public static Matrix operator *(Scalar s, Matrix m) => new Matrix(s.scalar * m.matrix);

        public static Matrix operator /(Scalar s, Matrix m) => new Matrix(s.scalar / m.matrix);

        #endregion S - M

        #region M - S

        public static Matrix operator +(Matrix m, Scalar s) => new Matrix(m.matrix + s.scalar);

        public static Matrix operator -(Matrix m, Scalar s) => new Matrix(m.matrix - s.scalar);

        public static Matrix operator *(Matrix m, Scalar s) => new Matrix(m.matrix * s.scalar);

        public static Matrix operator /(Matrix m, Scalar s) => new Matrix(m.matrix / s.scalar);

        #endregion M - S
    }
}