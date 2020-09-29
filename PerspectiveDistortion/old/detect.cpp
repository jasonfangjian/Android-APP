

#include <iostream>  
#include<vector>
#include<algorithm>
#include <opencv2\opencv.hpp>  
#include <opencv2\highgui\highgui.hpp>  
using namespace std;
using namespace cv;


//�������������С��������
bool ascendSort(vector<Point> a, vector<Point> b) {
	return a.size() < b.size();

}

//�������������С��������
bool descendSort(vector<Point> a, vector<Point> b) {
	return a.size() > b.size();
}





void floodFillborder(const cv::Mat& binsrcImg, cv::Mat& bindstImg)
{
	const int nr = binsrcImg.rows;
	const int nc = binsrcImg.cols;
	Mat edge[4];
	edge[0] = binsrcImg.row(0);    //up
	edge[1] = binsrcImg.row(nr - 1); //bottom
	edge[2] = binsrcImg.col(0);    //left
	edge[3] = binsrcImg.col(nc - 1); //right

	std::vector<Point> edgePts;
	const int minLength = std::min(nr, nc) / 4;
	for (int i = 0; i<4; ++i)
	{
		std::vector<Point> line;
		Mat_<uchar>::const_iterator iter = edge[i].begin<uchar>();       //��ǰ����
		Mat_<uchar>::const_iterator nextIter = edge[i].begin<uchar>() + 1; //��һ������
		while (nextIter != edge[i].end<uchar>())
		{
			if (*iter == 255)
			{
				if (*nextIter == 255)
				{
					Point pt = iter.pos();
					if (i == 1)
						pt.y = nr - 1;
					if (i == 3)
						pt.x = nc - 1;

					edgePts.push_back(pt);
				}
			}
			++iter;
			++nextIter;
		}
	}

	for (int n = 0; n<edgePts.size(); ++n)
		floodFill(binsrcImg, edgePts[n], 0);//��ˮ��䷨
	binsrcImg.copyTo(bindstImg);
}


int main145() {
	Mat srcImage = imread("my1.jpg");
	Mat thresholdImage;
	Mat grayImage;
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	//bitwise_not(grayImage, grayImage);
	threshold(grayImage, thresholdImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	//imshow("2", thresholdImage);
	//Mat resultImage;
	//thresholdImage.copyTo(resultImage);
	vector< vector< Point> > contours;  //���ڱ�������������Ϣ
	vector< vector< Point> > contours2; //���ڱ����������100������
	vector<Point> tempV;				//�ݴ������

	findContours(thresholdImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//cv::Mat labels;
	//int N = connectedComponents(resultImage, labels, 8, CV_16U);
	//findContours(labels, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	//�������������С������������
	sort(contours.begin(), contours.end(), ascendSort);//��������
	vector<vector<Point> >::iterator itc = contours.begin();
	int i = 0;
	while (itc != contours.end())
	{
		//��������ľ��α߽�
		Rect rect = boundingRect(*itc);
		int x = rect.x;
		int y = rect.y;
		int w = rect.width;
		int h = rect.height;
		//���������ľ��α߽�
		//cv::rectangle(srcImage, rect, { 0, 0, 255 }, 1);
		//����ͼƬ
		char str[10];
		sprintf(str, "%d.jpg", i++);
		//cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		//waitKey(1000);

		if (itc->size() < 200)
		{
			//�������������100�����򣬷ŵ�����contours2�У�
			tempV.push_back(Point(x, y));
			tempV.push_back(Point(x, y + h));
			tempV.push_back(Point(x + w, y + h));
			tempV.push_back(Point(x + w, y));
			contours2.push_back(tempV);
			/*Ҳ����ֱ���ã�contours2.push_back(*itc);���������5�����*/
			//contours2.push_back(*itc);

			//ɾ�������������100�����򣬼��ð�ɫ��������������100������
			cv::drawContours(srcImage, contours2, -1, Scalar(255,255,255), CV_FILLED);
		}
		//����ͼƬ
		sprintf(str, "%d.jpg", i++);
		//cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		//cv::waitKey(100);
		tempV.clear();
		++itc;
	}
	cv::imshow("srcImage", srcImage);

	Mat gray_src, bin_src, dst;

	cvtColor(srcImage, gray_src, CV_BGR2GRAY);
	bitwise_not(gray_src, gray_src);
	Mat thresImage = Mat::zeros(gray_src.rows, gray_src.cols, CV_8UC1);
	threshold(grayImage, thresImage, 250, 255, THRESH_BINARY);
	floodFillborder(thresImage, thresImage);
	bitwise_not(thresImage, thresImage);
	imshow("floodFill", thresImage);
	imwrite("floodfill.jpg", thresImage);
	waitKey(0);
	return 0;
}
