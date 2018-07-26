using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Animus.ANN.Core
{
    public class LayerParameterList : IDifferentiable<LayerParameterList>, IReadOnlyList<LayerParameters>
    {
        private readonly int offset;

        public LayerParameterList(IEnumerable<LayerParameters> parameters)
        {
            Raw = parameters.ToArray();

            offset = 0;

            Count = Raw.Length;
        }

        public LayerParameterList(LayerParameterList parent, int? s, int? e)
        {
            Raw = parent.Raw;

            offset = s ?? 0;

            Count = (e ?? (Raw.Length - 1)) - offset + 1;
        }

        public LayerParameters this[int index] => Raw[offset + index];

        public int Count { get; }

        public LayerParameters[] Raw { get; }

        public IEnumerator<LayerParameters> GetEnumerator()
        {
            foreach (var item in Raw)
            {
                yield return item;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            foreach (var item in Raw)
            {
                yield return item;
            }
        }

        #region differential

        public LayerParameterList Primal => new LayerParameterList(Raw.Select(lp => lp.Primal));

        public LayerParameterList Tangent => new LayerParameterList(Raw.Select(lp => lp.Tangent));

        public LayerParameterList Adjoint => new LayerParameterList(Raw.Select(p => p.Adjoint));

        public LayerParameterList Forward(LayerParameterList tangent, uint lvl) => new LayerParameterList(Raw.Select((p, idx) => p.Forward(tangent[idx], lvl)));

        public LayerParameterList Reverse(uint lvl) => new LayerParameterList(Raw.Select(p => p.Reverse(lvl)));

        #endregion differential
    }
}