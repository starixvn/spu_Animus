using Animus.Tensor;
using MathNet.Numerics.Statistics;
using System.Collections.Generic;
using System.Linq;

namespace Animus
{
    public static class Statistics
    {
        public static Vector Variance(this IEnumerable<Vector> data)
        {
            var average = Vector.Zero;
            var averageSq = Vector.Zero;

            foreach (var item in data)
            {
                average += item;
                averageSq += item.Pow(2);
            }

            var n = data.Count();

            return (averageSq - average.Pow(2)) / (n - 1);
        }

        public static Matrix Covariance(this IEnumerable<Vector> data)
        {
            var sumVector = Vector.Zero;
            var sumMatrix = Matrix.Zero;

            foreach (var item in data)
            {
                sumVector += item;
                sumMatrix += item.Cross(item);
            }

            var n = data.Count();

            return (sumMatrix - sumVector.Cross(sumVector)) / (n - 1);
        }

        public static Vector Mean(this IEnumerable<Vector> data)
        {
            var sum = Vector.Zero;

            foreach (var item in data)
            {
                sum += item;
            }

            return sum / data.Count();
        }

        public static double Median(this IEnumerable<double> data) => MathNet.Numerics.Statistics.Statistics.Median(data);

        public static double Skewness(this IEnumerable<double> data) => MathNet.Numerics.Statistics.Statistics.Skewness(data);

        public static Histogram Histogram(this IEnumerable<double> data, double min, double max, int bucket) => new Histogram(data, bucket, min, max);
    }
}