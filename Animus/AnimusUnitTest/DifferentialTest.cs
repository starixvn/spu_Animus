using Animus;
using Animus.Tensor;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AnimusUnitTest
{
    [TestClass]
    public class DifferentialTest
    {
        [TestMethod]
        public void TestDerivative()
        {
            var (functionValue, derivative) = Differential.Derivative(a => a * a, 5);

            Assert.AreEqual(5 * 5, (double)functionValue);
            Assert.AreEqual(2 * 5, (double)derivative);
        }

        [TestMethod]
        public void TestGradient()
        {
            var v = new Vector(1, 2, 3);
            var (functionValue, gradient) = Differential.Gradient(a => a * a, v);

            Assert.AreEqual((double)(v * v), (double)functionValue);

            for (int i = 0; i < v.Length; i++)
            {
                Assert.AreEqual((double)(2 * v[i]), (double)gradient[i]);
            }
        }
    }
}