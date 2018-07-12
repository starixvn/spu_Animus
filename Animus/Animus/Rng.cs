using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using System;
using System.Collections.Generic;

namespace Animus
{
    public static class Rng
    {
        public static Random Source { get; set; } = MersenneTwister.Default;

        public static ContinuousUniform Uniform(double min, double max) => new ContinuousUniform(min, max, Source);

        public static DiscreteUniform DiscreetUniform(int min, int max) => new DiscreteUniform(min, max, Source);

        public static Normal Normal(double mean, double stddev) => new Normal(mean, stddev, Source);

        public static Categorical Category(params double[] probabilities) => new Categorical(probabilities, Source);

        public static Bernoulli Bernoulli(double p) => new Bernoulli(p, Source);

        public static IEnumerable<T> SelectNoRepeat<T>(this IEnumerable<T> data, int n) => Combinatorics.SelectVariation(data, n, Source);

        public static IEnumerable<T> SelectWithRepeat<T>(this IEnumerable<T> data, int n) => Combinatorics.SelectVariationWithRepetition(data, n, Source);
    }
}