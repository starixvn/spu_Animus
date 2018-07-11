using Animus;
using Animus.Tensor;
using System;

namespace Test
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var a = new Matrix(new double[,] 
            {
                { 1, -2},
                { -2, 1}
            });

            Console.WriteLine(a + 5);

            Console.ReadLine();
        }
    }
}