#include "yolo8-cls.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	string model_path = argv[1], img_path = argv[2];
	int imgh = 224, imgw = 224;

	Mat input = imread(img_path), dst, blob;
	vector<double> results;

	dnn::Net model = dnn::readNetFromONNX(model_path);

	resize(input, dst, Size(imgw, imgh), 0, 0, INTER_LINEAR);
	dnn::blobFromImage(dst, blob, 1 / 255.0f, Size(dst.cols, dst.rows), Scalar(0, 0, 0), true, false);

	model.setInput(blob);
	
	vector<string> out_layer_names = model.getUnconnectedOutLayersNames();

	model.forward(results, out_layer_names);

	for (auto& result : results) {
		cout << result << endl;
	}
	
	return 0;
}
