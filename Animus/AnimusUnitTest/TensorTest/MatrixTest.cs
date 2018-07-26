using Animus.Tensor;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AnimusUnitTest.TensorTest
{
    [TestClass]
    public class MatrixTest
    {
        private Matrix FillMatrix(int m, int n, Random rng)
        {
            var arr = new double[m, n];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    arr[i, j] = rng.NextDouble();
                }
            }

            return arr;
        }

        [TestMethod]
        public void TestMMOperator()
        {
            var rng = new Random();

            var size = 5;

            var a = FillMatrix(size, size, rng);

            var b = FillMatrix(size, size, rng);

            var c = -a;
            var c1 = a + b;
            var c2 = a - b;
            var c3 = a * b;

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    var diff = Math.Abs((double)(c[i, j] + a[i, j]));
                    var diff1 = Math.Abs((double)(c1[i, j] - (a[i, j] + b[i, j])));
                    var diff2 = Math.Abs((double)(c2[i, j] - (a[i, j] - b[i, j])));

                    var sum = Enumerable.Range(0, size).Select(k => (double)(a[i, k] * b[k, j])).Sum();
                    var diff3 = Math.Abs(sum - (double)c3[i, j]);

                    Assert.IsTrue(diff < 1e-12);
                    Assert.IsTrue(diff1 < 1e-12);
                    Assert.IsTrue(diff2 < 1e-12);
                    Assert.IsTrue(diff3 < 1e-12);
                }
            }
        }
    }
}