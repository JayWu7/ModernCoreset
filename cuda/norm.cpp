#include "dataloader.h"
#include "dataloader.cpp"

using namespace std;
using namespace coreset;


int main(int argc, char **argv){
	if (argc != 3){
		std::cout << "Usage: ./norm <csv_file_path> <sep> \n";
   		 return EXIT_FAILURE;
	}
	DataLoader<float> dataloader;
	
	string csv_path = argv[1];
	char sep = argv[2][0];
	
	dataloader.NormCsv(csv_path, sep);	

	return 0;	
}
