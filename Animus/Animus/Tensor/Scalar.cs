using static DiffSharp.AD.Float64;

namespace Animus.Tensor
{
    public class Scalar : IDifferentiable<Scalar>
    {
        internal D scalar;

        public Scalar(double scalar) => this.scalar = D.NewD(scalar);

        internal Scalar(D scalar) => this.scalar = scalar;

        public override string ToString() => $"{(double)scalar}";

        public static Scalar Zero { get; } = new Scalar(0);

        public static Scalar One { get; } = new Scalar(1);

        #region instance operator

        public Scalar Pow(Scalar s) => new Scalar(D.Pow(scalar, s.scalar));

        #endregion instance operator

        #region differential

        public Scalar Primal => new Scalar(scalar.P);

        public Scalar Tagent => new Scalar(scalar.T);

        public Scalar Adjoint => new Scalar(scalar.A);

        public Scalar Forward(Scalar tangent, uint lvl)
        {
            var result = scalar.GetForward(tangent.scalar, lvl);

            return new Scalar(result);
        }

        public Scalar Reverse(uint lvl)
        {
            var result = scalar.GetReverse(lvl);

            return new Scalar(result);
        }

        #endregion differential

        #region casting

        public static implicit operator Scalar(double a) => new Scalar(a);

        public static explicit operator double(Scalar s) => (double)s.scalar;

        #endregion casting

        #region s - s

        public static Scalar operator -(Scalar a) => new Scalar(-a.scalar);

        public static Scalar operator +(Scalar a, Scalar b) => new Scalar(a.scalar + b.scalar);

        public static Scalar operator -(Scalar a, Scalar b) => new Scalar(a.scalar - b.scalar);

        public static Scalar operator *(Scalar a, Scalar b) => new Scalar(a.scalar * b.scalar);

        public static Scalar operator /(Scalar a, Scalar b) => new Scalar(a.scalar / b.scalar);

        #endregion s - s
    }
}