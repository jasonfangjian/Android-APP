#include <iostream>
#include <string>
#include <sstream>
#include <array>
#include <tuple>
// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "fm_ocr_scanner.hpp"

#include <math.h>
using namespace std;
using namespace cv;


Mat cutImage(Mat &src, std::vector<cv::Point> cornors4) {
	//�����ǵ�
	float widthRate = src.cols/256.0;
	float heightRate = src.rows / 256.0;
	//�ü�����
	const int indent = 2;
	//4 corners: top - left, top - right, bottom - right, bottom - left, index are 0, 1, 2, 3
	cv::Point2f srcVertex[4], dstVertex[4];
	srcVertex[0].x = cornors4[0].x*widthRate;
	srcVertex[0].y = cornors4[0].y*heightRate;
	srcVertex[1].x = cornors4[1].x*widthRate;
	srcVertex[1].y = cornors4[1].y*heightRate;
	srcVertex[2].x = cornors4[3].x*widthRate;
	srcVertex[2].y = cornors4[3].y*heightRate;
	srcVertex[3].x = cornors4[2].x*widthRate;
	srcVertex[3].y = cornors4[2].y*heightRate;

	dstVertex[0].x = cornors4[0].x*widthRate;
	dstVertex[0].y = cornors4[0].y*heightRate;
	dstVertex[1].x = cornors4[1].x*widthRate;
	dstVertex[1].y = cornors4[0].y*heightRate;
	dstVertex[2].x = cornors4[0].x*widthRate;
	dstVertex[2].y = cornors4[3].y*heightRate;
	dstVertex[3].x = cornors4[1].x*widthRate;
	dstVertex[3].y = cornors4[3].y*heightRate;
	//͸��
	Mat warp_mat = cv::getPerspectiveTransform(srcVertex, dstVertex);
	cv::Mat perspective;
	cv::warpPerspective(src, perspective, warp_mat, cv::Size(src.cols, src.rows), cv::INTER_LINEAR);
	//��ȡROI
	Rect rectInImage;
	int x0 = max(int(dstVertex[0].x - indent), 0);
	int y0 = max(int(dstVertex[0].y - indent), 0);
	int width = (srcVertex[1].x - srcVertex[0].x) + indent * 2;
	int height = (srcVertex[2].y - srcVertex[0].y) + indent * 2;
	if (x0 + width > src.cols)
		width = (srcVertex[1].x - srcVertex[0].x);
	if (y0 + height > src.rows)
		height = (srcVertex[2].y - srcVertex[0].y);
	rectInImage = Rect(x0, y0, width, height);//�ǵ�����
	Size size;
	size.width = rectInImage.width;
	size.height = rectInImage.height;
	//�ü�ROI
	Mat cutImg;
	Mat roi = perspective(rectInImage);
	roi.copyTo(cutImg);
	return cutImg;
}


int main()
{
	//ԭͼ
	Mat src0 = cv::imread("test/test8.jpg");	
	Mat fuse0 = cv::imread("test/test8_fuse.png");
	//����
	Mat src(src0.rows, src0.cols, CV_8UC1);
	resize(src0,src, fuse0.size());
	Mat fuse;
	cvtColor(fuse0, fuse, CV_RGB2GRAY);
	//����Ԫ��
	std::tuple<bool, std::vector<cv::Point>, std::vector<cv::Mat> > tup;
	//����
	tup = ProcessEdgeImage(fuse, src, 1);
	//��ַ���ֵ
	bool jug;
	std::vector<cv::Point> rect;
	std::vector<cv::Mat> imgs;
	std::tie(jug, rect, imgs) = tup;
	//�ж��Ƿ���ȡ�ɹ�
	if (!jug) {
		printf("failed!");
		return 0;
	}
	//����
	Mat cutImg = cutImage(src0, rect);
	//�������
	imshow("fuse", imgs[0]);
	imshow("lines", imgs[1]);
	imshow("corners", imgs[2]);
	imshow("rect", imgs[3]);
	imshow("cut", cutImg);
	waitKey();

	return 0;
}
