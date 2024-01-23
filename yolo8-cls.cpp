#include "yolo8-cls.h"

using namespace std;
using namespace cv;

void infer(string model_path, string img_path) {
	int imgh = 224, imgw = 224;

	Mat input = imread(img_path), dst;
	vector<double> results;

	dnn::Net model = dnn::readNetFromONNX(model_path);

	resize(input, dst, Size(imgw, imgh), 0, 0, INTER_LINEAR);
	dnn::blobFromImage(dst, input, 1 / 255.0f, Size(dst.cols, dst.rows), Scalar(0, 0, 0), true, false);

	model.setInput(input);
	results = model.forward();

	if (results[0] > results[1]) cout << "off" << endl;
	else cout << "on" << endl;
}

int main(int argc, char *argv[])
{
	string model_path = argv[1], img_path = argv[2];
	infer(model_path, img_path);

	return 0;
}
