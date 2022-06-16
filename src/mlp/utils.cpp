
#include <iostream>
#include <utility>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

vector<vector<double>> read_eigenvectors()
{
    string fname = "../data/output/imageProcessingOutput/eigenvectors.csv";

    vector<vector<double>> data;
    vector<double> row;
    string line, word;

    fstream file(fname, ios::in);
    if (file.is_open())
    {
        while (getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while (getline(str, word, ','))
                row.push_back(stod(word));
            data.push_back(row);
        }
    }
    else
        cout << "Could not open the file\n";
    return data;
}

int main()
{
    auto a = read_eigenvectors();
    cout << "Size: " << a[0].size() << "\n";
    cout << "First value: " << a[0][0] << "\n";
    return 0;
}