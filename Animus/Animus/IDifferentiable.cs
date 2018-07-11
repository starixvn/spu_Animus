namespace Animus
{
    public interface IDifferentiable<T>
    {
        T Primal { get; }

        T Tangent { get; }

        T Adjoint { get; }

        T Forward(T tangent, uint lvl);

        T Reverse(uint lvl);
    }
}