#include <iostream>
using namespace std;

int f(int x, int y){
    return x+y;
}



int main(){
    // cout<<"hello world"<<endl;
    // int x = 3;
    // cout<<x<<endl;

    for(int i=0; i<=3; i++){
        cout<<i<<endl;
    }

    int i = 0;
    while (i<=4){
        cout<<i<<endl;
        i+=1;
    }

    cout<<f(10,20)<<endl;

    int x = 3;
    if (x <=3){
        cout<"x<=3"<<endl;
    }
    else if(x <=10){
        cout<""
    }

    return 0;
}