#include <opencv2/opencv.hpp>
using namespace cv;
#pragma comment(lib, "opencv_world331.lib")

#include <iostream>
using namespace std;

int main(void)
{
	Mat leftimg = imread("left.png", IMREAD_GRAYSCALE);
	Mat rightimg = imread("right.png", IMREAD_GRAYSCALE);
	Size imagesize = leftimg.size();
	Mat disparity = Mat(imagesize.height, imagesize.width, CV_16S);
	Mat disparity_float;

	Ptr<StereoBM> sbm = StereoBM::create();
	sbm->setDisp12MaxDiff(1);
	sbm->setSpeckleRange(8);
	sbm->setSpeckleWindowSize(50);
	sbm->setUniquenessRatio(0);
	sbm->setTextureThreshold(507);
	sbm->setMinDisparity(-30);
	sbm->setPreFilterCap(61);
	sbm->setPreFilterSize(5);
	sbm->compute(leftimg, rightimg, disparity);
	normalize(disparity, disparity_float, 0, 1, CV_MINMAX, CV_32FC1);

	imshow("left", leftimg);
	imshow("right", rightimg);
	imshow("disp", disparity_float);

	waitKey(0);

	destroyAllWindows();

	return 0;
}