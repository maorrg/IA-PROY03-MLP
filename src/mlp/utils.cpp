
#include <iostream>
#include <utility>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;


void read_eigenvectors()
{
    string fname = "../data/output/imageProcessingOutput/eigenvectors.csv";

    vector<vector<string> > content;
    vector<string> row;
    string line, word;

    fstream file(fname, ios::in);
    if (file.is_open())
    {
        while (getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while (getline(str, word, ','))
                row.push_back(word);
            content.push_back(row);
        }
    }
    else
        cout << "Could not open the file\n";

    for (int i = 0; i < content.size(); i++)
    {
        for (int j = 0; j < content[i].size(); j++)
        {
            cout << content[i][j] << " ";
        }
        cout << "\n";
    }

}


int main(){
    read_eigenvectors();
    return 0;
}