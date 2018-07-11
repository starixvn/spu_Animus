using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Animus
{
    public static class TaskExtension
    {
        public static async Task<R> Map<T, R>(this Task<T> task, Func<T, R> func)
        {
            var result = await task;

            return func(result);
        }

        public static Task<T> Lift<T>(this T data) => new Task<T>(() => data);

        public static async Task<IEnumerable<T>> Collect<T>(this IEnumerable<Task<T>> tasks)
        {
            var arr = tasks.ToArray();

            await Task.WhenAll(tasks);

            return tasks.Select(t => t.Result);
        }

        public static async void Collect(this IEnumerable<Task> tasks) => await Task.WhenAll(tasks);
    }
}