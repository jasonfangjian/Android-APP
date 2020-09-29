#include "stdio.h"  
#include "cv.h"  
#include "highgui.h"  
#include "Math.h"

#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
using namespace std;
using namespace cv;

/*

class CPerspective
{
private:
	vector<Vec4i> lines;
	double deltaRho;
	double deltaTheta;
	int minVote;
	double minLength;
	double maxGap;

public:
	CPerspective() : deltaRho(1), deltaTheta(CV_PI / 180), minVote(10), minLength(0.), maxGap(0.) {}

	void setAccResolution(double dRho, double dTheta)
	{
		deltaRho = dRho;
		deltaTheta = dTheta;
	}

	void setMinVote(int minv)
	{
		minVote = minv;
	}

	void setLineLengthAndGap(double length, double gap)
	{
		minLength = length;
		maxGap = gap;
	}

	std::vector<Vec4i> findLines(Mat& binary)
	{
		lines.clear();
		HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);
		return lines;
	}

	vector<Vec4i> drawVerticalLines(Mat &image, Scalar color = Scalar(0, 0, 255))
	{
		vector<Vec4i>::const_iterator iter = lines.begin();
		vector<Vec4i> verline;

		while (iter != lines.end())
		{
			Point pt1((*iter)[0], (*iter)[1]);
			Point pt2((*iter)[2], (*iter)[3]);

			if (pt1.x == pt2.x || (pt2.y - pt1.y) / (pt2.x - pt1.x)>1 || (pt2.y - pt1.y) / (pt2.x - pt1.x)<-1)
			{
				//line(image, pt1, pt2, color);  
				verline.push_back(*iter);
				++iter;
			}
			else
				++iter;
		}

		return verline;
	}

	vector<Vec4i> drawHorizontalLines(Mat &image, Scalar color = Scalar(0, 0, 255))
	{
		vector<Vec4i>::const_iterator iter1 = lines.begin();
		vector<Vec4i>::const_iterator iter2 = lines.begin();
		vector<Vec4i>::const_iterator top_iter = iter1;
		vector<Vec4i>::const_iterator bottom_iter = iter1;
		vector<Vec4i> horline;

		int miny = (*iter1)[1], maxy = (*iter1)[1];
		while (iter1 != lines.end())
		{
			if (miny > (*iter1)[1])
			{
				miny = (*iter1)[1];
				top_iter = iter1;
				iter1++;
			}
			else
				iter1++;
		}
		Point pt_top1((*top_iter)[0], (*top_iter)[1]);
		Point pt_top2((*top_iter)[2], (*top_iter)[3]);
		//line(image, pt_top1, pt_top2, color);   
		horline.push_back(*top_iter);

		while (iter2 != lines.end())
		{
			if (maxy < (*iter2)[1])
			{
				maxy = (*iter2)[1];
				bottom_iter = iter2;
				iter2++;
			}
			else
				iter2++;
		}
		Point pt_bottom1((*bottom_iter)[0], (*bottom_iter)[1]);
		Point pt_bottom2((*bottom_iter)[2], (*bottom_iter)[3]);
		//line(image, pt_bottom1, pt_bottom2, color); 
		horline.push_back(*bottom_iter);

		return horline;
	}

	static CvPoint computeIntersect(Vec4i a, Vec4i b)
	{
		int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
		int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

		if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
		{
			CvPoint pt;
			pt.x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d;
			pt.y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d;
			return pt;
		}
	}
};

IplImage* mycvPerspectiveCorrectionImage(Mat& image)
{
	Mat cannyMat;
	Canny(image, cannyMat, 60, 220, 3);

	CPerspective findVertex;
	findVertex.setMinVote(90);
	findVertex.setLineLengthAndGap(100, 80);
	findVertex.findLines(cannyMat);
	vector<Vec4i> horline = findVertex.drawVerticalLines(image);
	vector<Vec4i> verline = findVertex.drawHorizontalLines(image);

	CvPoint2D32f srcVertex[4], dstVertex[4];
	srcVertex[0].x = CPerspective::computeIntersect(horline[1], verline[0]).x;
	srcVertex[0].y = CPerspective::computeIntersect(horline[1], verline[0]).y;
	srcVertex[1].x = CPerspective::computeIntersect(horline[0], verline[0]).x;
	srcVertex[1].y = CPerspective::computeIntersect(horline[0], verline[0]).y;
	srcVertex[2].x = CPerspective::computeIntersect(horline[1], verline[1]).x;
	srcVertex[2].y = CPerspective::computeIntersect(horline[1], verline[1]).y;
	srcVertex[3].x = CPerspective::computeIntersect(horline[0], verline[1]).x;
	srcVertex[3].y = CPerspective::computeIntersect(horline[0], verline[1]).y;

	dstVertex[0].x = 0;
	dstVertex[0].y = 0;
	dstVertex[1].x = image.cols - 1;
	dstVertex[1].y = 0;
	dstVertex[2].x = 0;
	dstVertex[2].y = image.rows - 1;
	dstVertex[3].x = image.cols - 1;
	dstVertex[3].y = image.rows - 1;

	CvMat* warp_mat = cvCreateMat(3, 3, CV_32FC1);
	cvGetPerspectiveTransform(srcVertex, dstVertex, warp_mat);

	IplImage* srcImg = &IplImage(image);
	IplImage* dstImg = cvCloneImage(srcImg);
	cvWarpPerspective(srcImg, dstImg, warp_mat, CV_INTER_LINEAR, cvScalarAll(255));

	cvNamedWindow( "Perspective Warp" );  
	cvShowImage( "Perspective Warp", dstImg );  //×îÖÕÊÇÊä³ödst  

	return dstImg;
}


int main() {
	Mat image, result;
	image = imread("cut.jpg");
	mycvPerspectiveCorrectionImage(image);
	waitKey();
	return 0;
}

*/