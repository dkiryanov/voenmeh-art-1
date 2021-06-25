namespace Art
{
    public class Matrix
    {
        private readonly double[][] _matrix;

        public Matrix(int rows, int cols)
        {
            _matrix = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                _matrix[i] = new double[cols];
            }
        }

        public double this[int row, int col]
        {
            get => _matrix[row][col];
            set => _matrix[row][col] = value;
        }

        public double[] GetColumn(int columnIndex)
        {
            double[] columnData = new double[RowsCount];

            for (int i = 0; i < RowsCount; i++)
            {
                columnData[i] = _matrix[i][columnIndex];
            }

            return columnData;
        }

        public double[] GetRow(int index)
        {
            return _matrix[index];
        }

        public int ColumnsCount => _matrix[0].Length;

        public int RowsCount => _matrix.GetUpperBound(0) + 1;
    }
}