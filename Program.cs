using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Art
{
    class Program
    {
        public const int InputNeuronsCount = 12;
        public const int OutputNeuronsCount = 5;

        private static readonly List<List<int>> _images =
            new List<List<int>>(){
            new List<int>() { 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 },
            new List<int>() { 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1 },
            new List<int>() { 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1 },
            new List<int>() { 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 },
            new List<int>() { 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0 }
        };

        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.Title = "ЗМИ50108 Кирьянов Д. А. Обучение нейронной сети АРТ-1";

            Art1 network = new Art1(InputNeuronsCount, OutputNeuronsCount)
            {
                Vigilance = 0.85
            };

            for (int i = 0; i < _images.Count(); i++)
            {
                Console.WriteLine($"На вход подается изображение №{i+1}\n");
                Console.WriteLine($"Кодировка изображения: '{string.Join(" ", _images[i])}'\n");

                DataModel dataIn = new DataModel(_images[i]);
                DataModel dataOut = new DataModel(OutputNeuronsCount);

                Matrix inputWeights = network.InputLayerWeights;

                Console.WriteLine("Веса слоя сравнения (входной слой):");
                for (int columnIndex = 0; columnIndex < inputWeights.ColumnsCount; columnIndex++)
                {
                    Console.WriteLine($"T{columnIndex + 1}: {string.Join(" ", inputWeights.GetColumn(columnIndex))}");
                }

                Matrix outputWeights = network.OutputLayerWeights;

                Console.WriteLine("\nВеса слоя распознавания (выходной слой):");
                for (int rowIndex = 0; rowIndex < outputWeights.RowsCount; rowIndex++)
                {
                    Console.WriteLine($"B{rowIndex + 1}: {string.Join(" ", outputWeights.GetRow(rowIndex).Select(x => Math.Round(x, 2)))}");
                }

                network.Compute(dataIn, dataOut);

                Console.WriteLine("\nВыходные значения Sj нейронов слоя распознавания");
                Console.WriteLine("Нейрон:\t 1\t 2\t 3\t 4\t 5\t");
                Console.WriteLine($"    Sj:\t {string.Join("\t ", network.NeuronOutputs.Select(x => Math.Round(x, 2)))}");

                Console.WriteLine($"\nПобедил нейрон №{network.WinnerNeuron + 1}. Sн = {Math.Round(network.WinnerNeuronOutput, 2)}");
                Console.WriteLine($"Изображение относится к классу {network.WinnerNeuron + 1}");

                Console.WriteLine("---------------------------------------------------------------\n");
            }

            Console.WriteLine("Состояние нейронной сети после завершения обучения\n");
            Console.WriteLine("Веса слоя сравнения (входной слой):");
            for (int columnIndex = 0; columnIndex < network.InputLayerWeights.ColumnsCount; columnIndex++)
            {
                Console.WriteLine($"T{columnIndex + 1}: {string.Join(" ", network.InputLayerWeights.GetColumn(columnIndex))}");
            }

            Console.WriteLine("\nВеса слоя распознавания (выходной слой):");
            for (int rowIndex = 0; rowIndex < network.OutputLayerWeights.RowsCount; rowIndex++)
            {
                Console.WriteLine($"B{rowIndex + 1}: {string.Join(" ", network.OutputLayerWeights.GetRow(rowIndex).Select(x => Math.Round(x, 2)))}");
            }

            Console.WriteLine("\nВыходные значения Sj нейронов слоя распознавания");
            Console.WriteLine("Нейрон:\t 1\t 2\t 3\t 4\t 5\t");
            Console.WriteLine($"    Sj:\t {string.Join("\t ", network.NeuronOutputs.Select(x => Math.Round(x, 2)))}");

            Console.WriteLine("---------------------------------------------------------------\n");

            Console.WriteLine("\nОбучение нейросети закончено. Нажмите 'пробел' для выхода...");


            Console.ReadKey();
        }
    }
}
