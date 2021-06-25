using System.Collections.Generic;

namespace Art
{
    public class Art1
    {
        private readonly bool[] _memorizedOutputLayerNeurons;

        /// <summary>
        /// Данные слоя сравнения (входной слой)
        /// </summary>
        private readonly DataModel _inputLayerData;

        /// <summary>
        /// Данные слоя распознавания (выходной слой)
        /// </summary>
        private readonly DataModel _outputLayerData;

        /// <summary>
        /// Создает нейронную сеть АРТ-1
        /// </summary>
        /// <param name="inputLayerNeuronsCount">Количество нейронов для входного слоя</param>
        /// <param name="outputLayerNeuronsCount">Количество нейронов для выходного слоя</param>
        public Art1(int inputLayerNeuronsCount, int outputLayerNeuronsCount)
        {
            L = 2.5;
            Vigilance = 0.8d;

            InputLayerWeights = new Matrix(inputLayerNeuronsCount, outputLayerNeuronsCount);
            OutputLayerWeights = new Matrix(outputLayerNeuronsCount, inputLayerNeuronsCount);
            NeuronOutputs = new double[outputLayerNeuronsCount];

            _memorizedOutputLayerNeurons = new bool[outputLayerNeuronsCount];

            _inputLayerData = new DataModel(inputLayerNeuronsCount);
            _outputLayerData = new DataModel(outputLayerNeuronsCount);

            NotWinnerNeuron = outputLayerNeuronsCount;
            ActivationThreshold = 0.5;

            ResetMatrixWeights();
        }

        public double WinnerNeuronOutput { get; private set; }

        public double[] NeuronOutputs { get; private set; }

        /// <summary>
        /// Устанавливает параметр L
        /// </summary>
        public double L { get; set; }

        /// <summary>
        /// Устанавливает пороговое значение для активационной функции
        /// </summary>
        public double ActivationThreshold { get; set; }

        /// <summary>
        /// Значение, которое возвращается, если не бы найден нейрон-победитель.  
        /// Данное значение равно количеству нейронов выходного слоя + 1
        /// </summary>
        public int NotWinnerNeuron { get; private set; }

        /// Нейрон-победитель
        public int WinnerNeuron { get; private set; }

        /// <summary>
        /// Определяет факт получения нейрона-победителя в процессе обучения
        /// </summary>
        public bool HasWinnerNeuron => WinnerNeuron != NotWinnerNeuron;

        /// <summary>
        /// Параметр сходства 
        /// </summary>
        public double Vigilance { get; set; }

        /// <summary>
        /// Веса входного слоя
        /// </summary>
        public Matrix InputLayerWeights { get; set; }

        /// <summary>
        /// Веса выходного слоя
        /// </summary>
        public Matrix OutputLayerWeights { get; set; }

        /// <summary>
        /// Производит классификацию данных
        /// </summary>
        /// <param name="input">Входные данные</param>
        /// <returns>Класс, к которому принадлежат входные данные</returns>
        public int Classify(DataModel input)
        {
            DataModel inputData = new DataModel(new List<int>(_inputLayerData.Count));
            DataModel outputData = new DataModel(new List<int>(_outputLayerData.Count));

            for (int i = 0; i < inputData.Count; i++)
            {
                inputData.SetItem(i, input.GetItem(i));
            }

            Compute(inputData, outputData);

            return HasWinnerNeuron ? WinnerNeuron : -1;
        }

        /// <summary>
        /// Вычисляет выход нейронной сети АРТ-1
        /// </summary>
        /// <param name="input">Входные данные</param>
        /// <param name="output">Выходные данные</param>
        public void Compute(DataModel input, DataModel output)
        {
            for (int i = 0; i < _outputLayerData.Count; i++)
            {
                _memorizedOutputLayerNeurons[i] = false;
            }

            bool isResonance = false;
            bool isExhausted = false;

            do
            {
                SetInputLayer(input);
                ComputeOutputLayer();
                SetOutputLayer(output);

                if (WinnerNeuron != NotWinnerNeuron)
                {
                    ComputeInputLayer(input);

                    double signal = Magnitude(_inputLayerData) / Magnitude(input);
                
                    if (signal >= Vigilance)
                    {
                        isResonance = true;
                    }
                    else
                    {
                        _memorizedOutputLayerNeurons[WinnerNeuron] = true;
                    }
                }
                else
                {
                    isExhausted = true;
                }
            } while (!(isResonance || isExhausted));

            InitializeWeightsWithDefaultValues();
        }

        /// <summary>
        /// Вычисляет выход слоя распознавания (входной слой)
        /// </summary>
        private void ComputeInputLayer(DataModel input)
        {
            for (int i = 0; i < input.Count; i++)
            {
                // вычисляем свертку весов
                double sum = InputLayerWeights[i, WinnerNeuron]
                             * _outputLayerData.GetItem(WinnerNeuron);

                // Вычисляем значение активационной функции
                double activation = (input.GetItem(i) + sum) 
                                    / (1 + (input.GetItem(i) + sum));

                _inputLayerData.SetItem(i, activation > ActivationThreshold ? 1 : 0);
            }
        }

        /// <summary>
        /// Вычисляет выход слоя сравнения (выходной слой)
        /// </summary>
        private void ComputeOutputLayer()
        {
            WinnerNeuronOutput = -1d;
            WinnerNeuron = NotWinnerNeuron;

            for (int i = 0; i < _outputLayerData.Count; i++)
            {
                if (!_memorizedOutputLayerNeurons[i])
                {
                    double sum = 0;

                    // Каждый нейрон в слое распознавания вычисляет свертку 
                    // вектора его весов и вектора–выхода нейронов слоя сравнения
                    for (int j = 0; j < _inputLayerData.Count; j++)
                    {
                        sum += OutputLayerWeights[i, j] * _inputLayerData.GetItem(j);
                    }

                    NeuronOutputs[i] = sum;

                    if (sum > WinnerNeuronOutput)
                    {
                        WinnerNeuronOutput = sum;
                        WinnerNeuron = i;
                    }
                }

                _outputLayerData.SetItem(i, 0);
            }

            // Найден нейрон-победитель?
            if (WinnerNeuron != NotWinnerNeuron)
            {
                _outputLayerData.SetItem(WinnerNeuron, 1); 
            }
        }

        /// <summary>
        /// Копирует данные выходного слоя в переданный объект
        /// </summary>
        /// <param name="output">Объект, в который необходимо скопировать 
        /// данные выходного слоя нейросети</param>
        private void SetOutputLayer(DataModel output)
        {
            for (int i = 0; i < _outputLayerData.Count; i++)
            {
                output.SetItem(i, _outputLayerData.GetItem(i));
            }
        }

        /// <summary>
        /// Вычисляет сумму компонент, составляющих заданный вектор
        /// </summary>
        private double Magnitude(DataModel input)
        {
            double result = 0;

            for (int i = 0; i < input.Count; i++)
            {
                result += input.GetItem(i);
            }

            return result;
        }

        /// <summary>
        /// Производит установку весов для нейронов входного и выходного слоя 
        /// </summary>
        private void InitializeWeightsWithDefaultValues()
        {
            for (int i = 0; i < _inputLayerData.Count; i++)
            {
                if (_inputLayerData.GetItem(i) == 1)
                {
                    double magnitudeInput = Magnitude(_inputLayerData);
                    InputLayerWeights[i, WinnerNeuron] = 1;
                    OutputLayerWeights[WinnerNeuron, i] = L / (L - 1 + magnitudeInput);
                }
                else
                {
                    InputLayerWeights[i, WinnerNeuron] = 0;
                    OutputLayerWeights[WinnerNeuron, i] = 0;
                }
            }
        }

        /// <summary>
        /// Сбрасывает веса матрицы на первоначальные значения
        /// </summary>
        private void ResetMatrixWeights()
        {
            for (int i = 0; i < _inputLayerData.Count; i++)
            {
                for (int j = 0; j < _outputLayerData.Count; j++)
                {
                    InputLayerWeights[i, j] = 1;
                    OutputLayerWeights[j, i] = L / (L - 1 + _inputLayerData.Count);
                }
            }
        }

        private void SetInputLayer(DataModel value)
        {
            for (int i = 0; i < _inputLayerData.Count; i++)
            {
                double activation = value.GetItem(i) / 1 + value.GetItem(i);

                _inputLayerData.SetItem(i, activation > ActivationThreshold ? 1 : 0);
            }
        }
    }
}