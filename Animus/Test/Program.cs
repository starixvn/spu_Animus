using Animus;
using Animus.Tensor;
using System;

namespace Test
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            Func<Vector, Scalar> f = v => v * v;

            var result = f.Gradient(new Vector(2, 5));

            Console.WriteLine(result.functionValue);

            Console.WriteLine(result.gradient);

            Console.ReadLine();
        }
    }

    public class A : IDifferentiable<A>
    {
        public Scalar s { get; set; }

        public Scalar s1 { get; set; }

        public A Primal => new A { s = s.Primal, s1 = s1.Primal };

        public A Tangent => throw new NotImplementedException();

        public A Adjoint => throw new NotImplementedException();

        public A Forward(A tangent, uint lvl)
        {
            throw new NotImplementedException();
        }

        public A Reverse(uint lvl)
        {
            throw new NotImplementedException();
        }
    }
}