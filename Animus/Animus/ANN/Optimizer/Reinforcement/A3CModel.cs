using Animus.ANN.Core;
using Animus.Tensor;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Optimizer.Reinforcement
{
    public class A3CModel<TState, TAction> : IModel<TState, IEnumerable<(TAction action, double score)>>
    {
        private readonly LayerParameterList parameters;
        private readonly IDataFlow<TState, Vector> flow;
        private readonly IEnvironment<TState, TAction> environment;
        private readonly A3C<TState, TAction> a3C;

        public A3CModel(LayerParameterList parameters, IDataFlow<TState, Vector> flow, IEnvironment<TState, TAction> environment, A3C<TState, TAction> a3c)
        {
            Parameters = parameters.Raw;
            this.parameters = parameters;
            this.flow = flow;
            this.environment = environment;
            a3C = a3c;
        }

        public LayerParameters[] Parameters { get; }

        public IEnumerable<(TAction action, double score)> Compute(TState input)
        {
            var valuePolicy = flow.Model(parameters, input);

            var actions = environment.GetActions(input);

            var policy = actions
                .Select(a => valuePolicy[a3C.ActionIndex(a) + 1]);

            var normalizedPolicy = TensorFunction.Softmax(Vector.OfArray(policy.ToArray()));

            foreach (var action in actions)
            {
                yield return (action, (double)normalizedPolicy[a3C.ActionIndex(action) + 1]);
            }
        }
    }
}