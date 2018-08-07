using Animus;
using Animus.ANN;
using Animus.ANN.Optimizer.Reinforcement;
using Animus.ANN.Types;
using Animus.ANN.Types.Layers;
using Animus.Tensor;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AnimusUnitTest.ANNTest
{
    [TestClass]
    public class ReinforcementTest
    {
        public const int size = 5;

        public const int traps = 2;

        [TestMethod]
        public async Task TestMaze()
        {
            var ff = new FeedForwardNet<State>(
                    inputLayer: LayerFactory.MakeInputLayer<State>(s => s.Vector, size * size),
                    hiddenLayers: new[] { 16, 8 }.Select(LayerFactory.MakeTanhLayer),
                    outputLayer: LayerFactory.MakeLinearLayer(5));

            var env = new Playground();

            var controller = new GameController();

            var model = await controller.Train(env, ff);

            Solve(env, model);
        }

        public static void Solve(Playground pg, IModel<State, IEnumerable<ActionScore<Move>>> model)
        {
            var s = pg.CreateInitialState();

            var maze = new string[size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    var idx = i * size + j;

                    if (idx == s.Target.Index)
                    {
                        maze[i, j] = "1 ";
                    }
                    else if (s.Traps.Any(tr => tr.Index == idx))
                    {
                        maze[i, j] = "* ";
                    }
                    else if (s.Player.Index == idx)
                    {
                        maze[i, j] = "P ";
                    }
                    else
                    {
                        maze[i, j] = "0 ";
                    }
                }
            }

            var q = new Queue<Move>();

            while (!pg.IsTerminalState(s))
            {
                if (q.Count >= size * 2)
                {
                    Assert.Fail();

                    foreach (var item in q)
                    {
                        Console.WriteLine(item);
                    }

                    return;
                }

                var action = (Move)0;
                var score = .0;

                foreach (var item in model.Compute(s))
                {
                    if (score < item.Score)
                    {
                        action = item.Action;
                        score = item.Score;
                    }
                }

                Console.WriteLine($"{action} - {score}");

                s = pg.NextState(s, action);

                q.Enqueue(action);

                Draw(maze, action, s, pg);
            }

            Draw(maze);

            Assert.AreEqual(s.Target.Index, s.Player.Index);
        }

        public static void Draw(string[,] maze, Move move, State result, Playground pg)
        {
            if (pg.IsTerminalState(result))
            {
                return;
            }

            switch (move)
            {
                case Move.Left:
                    maze[result.Player.X, result.Player.Y] = "← ";

                    break;

                case Move.Right:
                    maze[result.Player.X, result.Player.Y] = "→ ";
                    break;

                case Move.Up:
                    maze[result.Player.X, result.Player.Y] = "↑ ";
                    break;

                case Move.Down:
                    maze[result.Player.X, result.Player.Y] = "↓ ";
                    break;

                default:
                    break;
            }
        }

        public static void Draw(string[,] maze)
        {
            for (int i = 0; i < maze.GetLength(0); i++)
            {
                for (int j = 0; j < maze.GetLength(1); j++)
                {
                    Console.Write(maze[i, j]);
                }

                Console.Write("\n");
            }
        }
    }

    public enum Move
    {
        Left = 0,
        Right = 1,
        Up = 2,
        Down = 3
    }

    public class Position
    {
        public int X { get; set; }

        public int Y { get; set; }

        public int Index => X * ReinforcementTest.size + Y;
    }

    public class State
    {
        public Position Target { get; set; }

        public Position Player { get; set; }

        public HashSet<Position> Traps { get; set; }

        public Vector Vector
        {
            get
            {
                var arr = new double[ReinforcementTest.size * ReinforcementTest.size];

                arr[Target.Index] = 0.5;

                arr[Player.Index] = 1;

                foreach (var item in Traps)
                {
                    arr[item.Index] = -1;
                }

                return new Vector(arr);
            }
        }
    }

    public class Playground : IEnvironment<State, Move>
    {
        private Position target;

        private HashSet<Position> traps;

        private IEnumerable<int> range;

        public Playground()
        {
            range = Enumerable.Range(0, ReinforcementTest.size * ReinforcementTest.size);

            var picked = range.SelectNoRepeat(1 + ReinforcementTest.traps).ToList();

            range = range.Where(a => !picked.Contains(a)).ToList();

            target = new Position { X = picked[0] / ReinforcementTest.size, Y = picked[0] % ReinforcementTest.size };

            picked.RemoveAt(0);

            traps = new HashSet<Position>(picked.Select(a => new Position { X = a / ReinforcementTest.size, Y = a % ReinforcementTest.size }));
        }

        public State CreateInitialState()
        {
            var picked = range.SelectNoRepeat(1).First();

            var player = new Position
            {
                X = picked / ReinforcementTest.size,
                Y = picked % ReinforcementTest.size
            };

            return new State { Target = target, Traps = traps, Player = player };
        }

        public IEnumerable<Move> GetActions(State s)
        {
            return new[] { Move.Left, Move.Right, Move.Up, Move.Down };
        }

        public bool IsTerminalState(State s)
        {
            return s.Player.Index == s.Target.Index
                || traps.Any(tr => tr.Index == s.Player.Index);
        }

        public State NextState(State s, Move a)
        {
            var clone = new State
            {
                Target = target,
                Traps = traps,
                Player = new Position { X = s.Player.X, Y = s.Player.Y }
            };

            var player = clone.Player;

            switch (a)
            {
                case Move.Left:
                    player.Y = System.Math.Max(0, player.Y - 1);
                    break;

                case Move.Right:
                    player.Y = System.Math.Min(ReinforcementTest.size - 1, player.Y + 1);
                    break;

                case Move.Up:
                    player.X = System.Math.Max(0, player.X - 1);
                    break;

                case Move.Down:
                    player.X = System.Math.Min(ReinforcementTest.size - 1, player.X + 1);
                    break;

                default:
                    break;
            }

            return clone;
        }
    }

    public class GameController : A3C<State, Move>
    {
        private readonly ReinforcementOptimizer optimizer = new ReinforcementOptimizer();

        public override IAsyncOptimizer Optimizer => optimizer;

        public override double Discount => 0.99;

        public override int Threads => 8;

        public override int ActionIndex(Move action) => (int)action;

        public override double Reward(State current, Move action, State result)
        {
            if (result.Target.Index == result.Player.Index)
            {
                return 1;
            }

            if (result.Traps.Any(tr => tr.Index == result.Player.Index))
            {
                return -1;
            }

            var totalStep = 2 * ReinforcementTest.size - 1;

            return -0.5 / totalStep;
        }

        public override bool Terminate(int thread, int game) => game == 300;

        public override void ThreadCallback(int thread, int game, double point)
        {
            //Console.WriteLine($"{thread} - {game} - {point}");
        }

        public override bool StopAndUpdate(int game, int turn) => false;
    }

    public class ReinforcementOptimizer : Rms
    {
        protected override double LearningRate(int game, int turn) => 0.003;
    }
}