#include <iostream>

using namespace std;

/*
    Generates 64x64 monochrome video sequence of an adge moving from left to 
    right.
    Values of pixels in the sequence are in [0 - 255] range.
*/

const int resolution = 64;

/*
    Parameters:
    fps
    length
    speed (m/sec)
    area
*/

int main() {
    int* row = new int[resolution][resolution];
    
    for(int i = 0; i < resolution; i++) {
        row[i] = 0;
    }

    delete[] row;

    return 0;
}
