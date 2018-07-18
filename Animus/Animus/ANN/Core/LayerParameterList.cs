using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Core
{
    public class LayerParameterList : IDifferentiable<LayerParameterList>, IReadOnlyList<LayerParameters>
    {
        private readonly LayerParameters[] parameters;

        private readonly int offset;

        public LayerParameterList(IEnumerable<LayerParameters> parameters)
        {
            this.parameters = parameters.ToArray();

            offset = 0;

            Count = this.parameters.Length;
        }

        public LayerParameterList(LayerParameterList parent, int? s, int? e)
        {
            parameters = parent.parameters;

            offset = s ?? 0;

            Count = (e ?? (parameters.Length - 1)) - offset + 1;
        }

        public LayerParameters this[int index] => parameters[offset + index];

        public int Count { get; }

        public LayerParameterList Primal => new LayerParameterList(parameters.Select(lp => lp.Primal));

        public LayerParameterList Tangent => new LayerParameterList(parameters.Select(lp => lp.Tangent));

        public LayerParameterList Adjoint => throw new NotImplementedException();

        public LayerParameterList Forward(LayerParameterList tangent, uint lvl)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<LayerParameters> GetEnumerator()
        {
            throw new NotImplementedException();
        }

        public LayerParameterList Reverse(uint lvl)
        {
            throw new NotImplementedException();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            throw new NotImplementedException();
        }
    }
}