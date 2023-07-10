// #include <iostream>
// using namespace std;

// int main(){
//     cout<<"Hello";
//     return 0;
// }

#include <iostream>
#include <unistd.h>

using namespace std;

int main() {

    cout << "Line 1..." << flush;

    usleep(500000);

    cout << "\nLine 2" << endl;

    cout << "Line 3" << endl ;

    return 0;
}