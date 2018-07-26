using System.Collections.Generic;
using System.Linq;
using static DiffSharp.AD.Float64;

namespace Animus.Tensor
{
    public class Vector : IDifferentiable<Vector>
    {
        internal DV vector;

        public Vector(params double[] args) => vector = DV.NewDV(args);

        internal Vector(DV v) => vector = v;

        public Scalar this[int idx] => new Scalar(vector[idx]);

        public int Length => vector.Length;

        public int MaxIndex => DV.MaxIndex(vector);

        public int MinIndex => DV.MinIndex(vector);

        public Vector Slice(int? start, int? end)
        {
            var s = start.HasValue ? start.Value : 0;
            var e = start.HasValue ? end.Value : vector.Length - 1;

            var result = vector.GetSlice(new Microsoft.FSharp.Core.FSharpOption<int>(s), new Microsoft.FSharp.Core.FSharpOption<int>(e));

            return new Vector(result);
        }

        public Vector Append(Vector v) => new Vector(DV.Append(vector, v.vector));

        public override string ToString() => string.Join(",", (double[])vector);

        public static Vector Zero { get; } = new Vector(DV.Zero);

        public static Vector OfArray(IEnumerable<Scalar> scalars) => new Vector(DV.OfArray(scalars.Select(s => s.scalar).ToArray()));

        #region instance operator

        public Vector Pow(Scalar a) => new Vector(DV.Pow(vector, a.scalar));

        public Vector Pow(Vector a) => new Vector(DV.Pow(vector, a.vector));

        public Matrix Cross(Vector v) => new Matrix(DV.op_AmpMultiply(vector, v.vector));

        public Vector Hadamard(Vector v) => new Vector(DV.op_DotMultiply(vector, v.vector));

        #endregion instance operator

        #region differential

        public Vector Primal => new Vector(vector.P);

        public Vector Tangent => new Vector(vector.T);

        public Vector Adjoint => new Vector(vector.A);

        public Vector Forward(Vector tangent, uint lvl)
        {
            var result = vector.GetForward(tangent.vector, lvl);

            return new Vector(result);
        }

        public Vector Reverse(uint lvl)
        {
            var result = vector.GetReverse(lvl);

            return new Vector(result);
        }

        #endregion differential

        #region casting

        public static implicit operator Vector(double[] v) => new Vector(v);

        public static explicit operator double[] (Vector v) => (double[])v.vector;

        #endregion casting

        #region v - v

        public static Vector operator -(Vector v) => new Vector(-v.vector);

        public static Vector operator +(Vector v1, Vector v2) => new Vector(v1.vector + v2.vector);

        public static Vector operator -(Vector v1, Vector v2) => new Vector(v1.vector - v2.vector);

        public static Scalar operator *(Vector v1, Vector v2) => new Scalar(v1.vector * v2.vector);

        #endregion v - v

        #region s - v

        public static Vector operator +(Scalar s, Vector v) => new Vector(s.scalar + v.vector);

        public static Vector operator -(Scalar s, Vector v) => new Vector(s.scalar - v.vector);

        public static Vector operator *(Scalar s, Vector v) => new Vector(s.scalar * v.vector);

        public static Vector operator /(Scalar s, Vector v) => new Vector(s.scalar / v.vector);

        #endregion s - v

        #region v - s

        public static Vector operator +(Vector v, Scalar s) => new Vector(s.scalar + v.vector);

        public static Vector operator -(Vector v, Scalar s) => new Vector(v.vector - s.scalar);

        public static Vector operator *(Vector v, Scalar s) => new Vector(s.scalar * v.vector);

        public static Vector operator /(Vector v, Scalar s) => new Vector(v.vector / s.scalar);

        #endregion v - s
    }
}