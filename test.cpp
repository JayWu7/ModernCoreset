#include <iostream>
#include <time.h>

using namespace std;

int main(){
   size_t a = 1000000000000000;
   double b = 0.0;
   clock_t start,end;    
   start = clock();
   for(int i=0; i<a; i++){
        b = 100*100.2 * i;
   }
   end = clock();
   cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
   return 1;
}   
