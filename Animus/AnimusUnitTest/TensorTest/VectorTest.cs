using Animus.Tensor;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AnimusUnitTest.TensorTest
{
    [TestClass]
    public class VectorTest
    {
        [TestMethod]
        public void SliceTest1()
        {
            var a = new Vector(1, 2, 8, 7, 9, 5);

            var b = a.Slice(null, 1);

            for (int i = 0; i < b.Length; i++)
            {
                Assert.AreEqual((double)a[i], (double)b[i]);
            }
        }

        [TestMethod]
        public void SliceTest2()
        {
            var a = new Vector(1, 2, 8, 7, 9, 5);

            var b = a.Slice(null, a.Length + 1);

            Assert.AreEqual(a.Length, b.Length);
        }

        [TestMethod]
        public void SliceTest3()
        {
            Assert.ThrowsException<IndexOutOfRangeException>(() =>
            {
                var a = new Vector(1, 2, 8, 7, 9, 5);

                var b = a.Slice(0, a.Length + 1);

                Console.WriteLine(b);
            });
        }

        [TestMethod]
        public void AppendTest()
        {
            var a = new Vector();

            Assert.ThrowsException<NullReferenceException>(() => a.Append(null));
        }

        [TestMethod]
        public void AppendTest1()
        {
            var a = new Vector();

            a.Append(Vector.Zero);

            Assert.AreEqual(0, a.Length);
        }

        [TestMethod]
        public void AppendTest3()
        {
            var a = new Vector(1, 2, 3);

            var b = new Vector(4, 5, 6);

            var c = a.Append(b);

            Assert.AreEqual(a.Length + b.Length, c.Length);

            for (int i = 0; i < a.Length; i++)
            {
                Assert.AreEqual((double)a[i], (double)c[i]);
            }

            for (int i = 0; i < b.Length; i++)
            {
                Assert.AreEqual((double)b[i], (double)c[i + a.Length]);
            }
        }

        [TestMethod]
        public void OfSequenceTest()
        {
            var a = new[] { 1, 2, 3 };

            var b = Vector.OfSequence(a.Select(x => (Scalar)x));

            for (int i = 0; i < a.Length; i++)
            {
                Assert.AreEqual(a[i], (double)b[i]);
            }
        }

        [TestMethod]
        public void TestPow()
        {
            var a = new Vector(1, 2, 3);

            var b = a.Pow(2.5);

            for (int i = 0; i < a.Length; i++)
            {
                var x = Math.Pow((double)a[i], 2.5);
                Assert.AreEqual(x, (double)b[i]);
            }
        }

        [TestMethod]
        public void TestPow2()
        {
            var rng = new Random();

            var source = Enumerable.Range(0, 3).Select((_) => rng.NextDouble());

            var a = new Vector(source.ToArray());

            var b = new Vector(source.ToArray());

            Console.WriteLine($"{a}\n{b}");

            var c = a.Pow(b);

            for (int i = 0; i < a.Length; i++)
            {
                var x = (double)a[i];
                var y = (double)b[i];

                var z = Math.Pow(x, y);

                Assert.AreEqual(z, (double)c[i]);
            }
        }

        [TestMethod]
        public void TestCross()
        {
            var rng = new Random();

            var source = Enumerable.Range(0, 3).Select((_) => rng.NextDouble());

            var a = new Vector(source.ToArray());

            var b = new Vector(source.ToArray());

            var c = a.Cross(b);

            for (int i = 0; i < a.Length; i++)
            {
                for (int j = 0; j < b.Length; j++)
                {
                    Assert.AreEqual((double)(a[i] * b[j]), (double)c[i, j]);
                }
            }
        }

        [TestMethod]
        public void TestHadamard()
        {
            var rng = new Random();

            var source = Enumerable.Range(0, 3).Select((_) => rng.NextDouble());

            var a = new Vector(source.ToArray());

            var b = new Vector(source.ToArray());

            var c = a.Hadamard(b);

            for (int i = 0; i < a.Length; i++)
            {
                Assert.AreEqual((double)(a[i] * b[i]), (double)c[i]);
            }
        }
    }
}