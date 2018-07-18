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
    }
}