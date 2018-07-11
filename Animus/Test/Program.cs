using Animus;
using Animus.Tensor;
using System;

namespace Test
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var a = new Scalar(5);

            Func<Scalar, Scalar> f = x => x + 4;

            Console.WriteLine(f.Derivative(0).derivative);

            Console.ReadLine();
        }
    }
}