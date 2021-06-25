using System.Collections.Generic;

namespace Art
{
    public class DataModel
    {
        private const int DefaultItemValue = -1;

        private readonly List<int> _data;

        public DataModel(int count)
        {
            _data = new List<int>(count);

            for (int i = 0; i < count; i++)
            {
                _data.Add(DefaultItemValue);
            }
        }

        public DataModel(List<int> data)
        {
            _data = data;
        }

        public int Count => _data.Count;

        public int GetItem(int index)
        {
            return _data[index];
        }

        public void SetItem(int index, int value)
        {
            _data[index] = value;
        }
    }
}