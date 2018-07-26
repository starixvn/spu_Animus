using Animus.Tensor;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AnimusUnitTest.TensorTest
{
    [TestClass]
    public class ScalarTest
    {
        [TestMethod]
        public void TestStaticFieldOne()
        {
            var a = Scalar.One;
            var b = Scalar.One;

            Assert.AreEqual(1, (double)a);
            Assert.IsTrue(ReferenceEquals(a, b));
        }

        [TestMethod]
        public void TestPow()
        {
            var a = (Scalar)8;

            Assert.AreEqual(64, (double)a.Pow(2));
        }

        [TestMethod]
        public void TestOperator()
        {
            var a = (Scalar)12;
            var b = (Scalar)8;

            Assert.AreEqual(-12, (double)(-a));
            Assert.AreEqual(20, (double)(a + b));
            Assert.AreEqual(4, (double)(a - b));
            Assert.AreEqual(12 * 8, (double)(a * b));
            Assert.AreEqual(12.0 / 8, (double)(a / b));
        }
    }
}