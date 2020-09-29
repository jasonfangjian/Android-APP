#include "com_example_pynixs_dodistortion_OpenCVUtilsOld.h"
#include <iostream>
#include <string>
#include <sstream>
// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

#include <Android/log.h>
#include <Android/asset_manager.h>
#include <Android/asset_manager_jni.h>
#include <android/bitmap.h>

#define TAG "com.example.pynixs.dodistortion.jni"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG ,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG ,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG ,__VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,TAG ,__VA_ARGS__)

#define BYTE unsigned char

using namespace std;
using namespace cv;


double angle(Point pt1, Point pt2, Point pt0) {// finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    double ratio;//边长平方的比
    ratio = (dx1*dx1 + dy1 * dy1) / (dx2*dx2 + dy2 * dy2);
    //if (ratio<0.8 || 1.2<ratio) {//根据边长平方的比过小或过大提前淘汰这个四边形，如果淘汰过多，调整此比例数字
    //	//? ? ? Log("ratio\n");
    //	return 1.0;//根据边长平方的比过小或过大提前淘汰这个四边形
    //}
    return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}


void findSquares(const Mat& gray0, vector<vector<Point> >& squares) {// returns sequence of squares detected on the gray0. the sequence is stored in the specified memory storage
    squares.clear();


#define N 6
    Mat pyr, gray1, timg;


    // down-scale and upscale the gray0 to filter out the noise
    pyrDown(gray0, pyr, Size(gray0.cols / 2, gray0.rows / 2));
    pyrUp(pyr, timg, gray0.size());
    vector<vector<Point> > contours;


    // try several threshold levels
    for (int l = 0; l < N; l++) {
        // hack: use Canny instead of zero threshold level.
        // Canny helps to catch squares with gradient shading
        //? ? ? if (l == 0 ) {//可试试不对l==0做特殊处理是否能在不影响判断正方形的前提下加速判断过程
        //? ? ? ? ? // apply Canny. Take the upper threshold from slider
        //? ? ? ? ? // and set the lower to 0 (which forces edges merging)
        //? ? ? ? ? Canny(timg, gray1, 0, thresh, 5);
        //? ? ? ? ? // dilate canny output to remove potential
        //? ? ? ? ? // holes between edge segments
        //? ? ? ? ? dilate(gray1, gray1, Mat(), Point(-1,-1));
        //? ? ? } else {
        // apply threshold if l!=0:
        //? ? ?tgray(x,y) = gray1(x,y) < (l+1)*255/N ? 255 : 0
        gray1 = timg >= (l + 1) * 255 / N;
        //? ? ? }


        // find contours and store them all as a list
        findContours(gray1, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);


        vector<Point> approx;


        // test each contour
        for (size_t i = 0; i < contours.size(); i++) {
            // approximate contour with accuracy proportional
            // to the contour perimeter
            approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);//0.02为将毛边拉直的系数，如果对毛边正方形漏检，可试试调大


            // square contours should have 4 vertices after approximation
            // relatively large area (to filter out noisy contours)
            // and be convex.
            // Note: absolute value of an area is used because
            // area may be positive or negative - in accordance with the
            // contour orientation
            if (approx.size() == 4 && isContourConvex(Mat(approx))) {
                double area;
                area = fabs(contourArea(Mat(approx)));
                if (10000.0<area && area<250000.0) {//当正方形面积在此范围内……，如果有因面积过大或过小漏检正方形问题，调整此范围。
                    //? ? ? ? ? ? ? ? ? printf("area=%lg\n",area);
                    double maxCosine = 0.0;


                    for (int j = 2; j < 5; j++) {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                        if (maxCosine == 1.0) break;// //边长比超过设定范围
                    }


                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    //if (maxCosine < 0.1)
                    {//四个角和直角相比的最大误差，可根据实际情况略作调整，越小越严格
                        squares.push_back(approx);
                        return;//检测到一个合格的正方形就返回
                        //? ? ? ? ? ? ? ? ? } else {
                        //? ? ? ? ? ? ? ? ? ? ? Log("Cosine>=0.1\n");
                    }
                }
            }
        }
    }
}


void drawSquares(Mat& img, vector<vector<Point> > &squares)
{
    for (size_t j = 0; j < squares.size(); j++)
    {
        for (size_t i = 1; i < squares[j].size(); i++)
        {
            cv::line(img, squares[j][i - 1], squares[j][i], Scalar(0, 0, 255), 1, 8);
            if (i == squares[j].size() - 1)
            {
                cv::line(img, squares[j][i], squares[j][0], Scalar(0, 0, 255), 1, 8);
            }
        }
    }
}




// ========== = 寻找最大边框 ========== =
int findLargestSquare(const vector<vector<cv::Point> >& squares, vector<cv::Point>& biggest_square)
{
    if (!squares.size()) return -1;


    int max_width = 0;
    int max_height = 0;
    int max_square_idx = 0;
    for (int i = 0; i < squares.size(); i++)
    {
        cv::Rect rectangle = boundingRect(Mat(squares[i]));
        if ((rectangle.width >= max_width) && (rectangle.height >= max_height))
        {
            max_width = rectangle.width;
            max_height = rectangle.height;
            max_square_idx = i;
        }
    }
    biggest_square = squares[max_square_idx];
    return max_square_idx;
}


/**
根据三个点计算中间那个点的夹角? ?pt1 pt0 pt2
*/
double getAngle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}


/**
点到点的距离
@param p1 点1
@param p2 点2
@return 距离
*/
double getSpacePointToPoint(cv::Point p1, cv::Point p2)
{
    int a = p1.x - p2.x;
    int b = p1.y - p2.y;
    return sqrt(a * a + b * b);
}


/**
两直线的交点


@param a 线段1
@param b 线段2
@return 交点
*/

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];


    if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
        return pt;
    }
    else
        return cv::Point2f(-1, -1);
}


/**
对多个点按顺时针排序


@param corners 点的集合
*/
void sortCorners(std::vector<cv::Point2f>& corners)
{
    if (corners.size() == 0) return;
    //先延 X轴排列
    cv::Point pl = corners[0];
    int index = 0;
    for (int i = 1; i < corners.size(); i++)
    {
        cv::Point point = corners[i];
        if (pl.x > point.x)
        {
            pl = point;
            index = i;
        }
    }
    corners[index] = corners[0];
    corners[0] = pl;


    cv::Point lp = corners[0];
    for (int i = 1; i < corners.size(); i++)
    {
        for (int j = i + 1; j<corners.size(); j++)
        {
            cv::Point point1 = corners[i];
            cv::Point point2 = corners[j];
            if ((point1.y - lp.y*1.0) / (point1.x - lp.x)>(point2.y - lp.y*1.0) / (point2.x - lp.x))
            {
                cv::Point temp = point1;
                corners[i] = corners[j];
                corners[j] = temp;
            }
        }
    }
}

vector<cv::Point> cornors4;

void processImage(Mat &mat)
{
    vector<vector<Point>>contours;

    Mat src_gray, filtered, edges, dilated_edges;

    //获取灰度图像
    src_gray = mat.clone();
    //滤波，模糊处理，消除某些背景干扰信息
    blur(src_gray, filtered, cv::Size(3, 3));
    //腐蚀操作，消除某些背景干扰信息
    erode(filtered, filtered, Mat(), cv::Point(-1, -1), 3, 1, 1);


    int thresh = 35;
    //边缘检测
    Canny(filtered, edges, thresh, thresh * 3, 3);
    //膨胀操作，尽量使边缘闭合
    dilate(edges, dilated_edges, Mat(), cv::Point(-1, -1), 3, 1, 1);


    //vector<vector<cv::Point> > contours;
    vector<vector<cv::Point> > squares, hulls;
    //寻找边框
    findContours(dilated_edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);


    vector<cv::Point> hull, approx;
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() > 0)
        {
            try
            {
                //边框的凸包
                convexHull(Mat(contours[i]), hull, false);
                //多边形拟合凸包边框(此时的拟合的精度较低)
                approxPolyDP(Mat(hull), approx, arcLength(Mat(hull), true)*0.02, true);
                //筛选出面积大于某一阈值的，且四边形的各个角度都接近直角的凸四边形
                if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 40000 &&
                    isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;
                    for (int j = 2; j < 5; j++)
                    {
                        double cosine = fabs(getAngle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    //角度大概72度
                    if (maxCosine < 0.3)
                    {
                        squares.push_back(approx);
                        hulls.push_back(hull);
                    }
                }
            }
            catch (...)
            {
                int i = 0;
            }
        }
    }


    vector<cv::Point> largest_square;
    //找出外接矩形最大的四边形
    int idex = findLargestSquare(squares, largest_square);
    if (largest_square.size() == 0 || idex == -1) return ;


    //找到这个最大的四边形对应的凸边框，再次进行多边形拟合，此次精度较高，拟合的结果可能是大于4条边的多边形
    //接下来的操作，主要是为了解决，证件有圆角时检测到的四个顶点的连线会有切边的问题
    hull = hulls[idex];
    approxPolyDP(Mat(hull), approx, 3, true);
    vector<cv::Point> newApprox;
    double maxL = arcLength(Mat(approx), true)*0.02;
    //找到高精度拟合时得到的顶点中 距离小于 低精度拟合得到的四个顶点 maxL的顶点，排除部分顶点的干扰
    for (cv::Point p : approx)
    {
        if (!(getSpacePointToPoint(p, largest_square[0]) > maxL &&
              getSpacePointToPoint(p, largest_square[1]) > maxL &&
              getSpacePointToPoint(p, largest_square[2]) > maxL &&
              getSpacePointToPoint(p, largest_square[3]) > maxL))
        {
            newApprox.push_back(p);
        }
    }
    //找到剩余顶点连线中，边长大于 2 * maxL的四条边作为四边形物体的四条边
    vector<Vec4i> lines;
    for (int i = 0; i < newApprox.size(); i++)
    {
        cv::Point p1 = newApprox[i];
        cv::Point p2 = newApprox[(i + 1) % newApprox.size()];
        if (getSpacePointToPoint(p1, p2) > 2 * maxL)
        {
            lines.push_back(Vec4i(p1.x, p1.y, p2.x, p2.y));
        }
    }


    //计算出这四条边中 相邻两条边的交点，即物体的四个顶点
    vector<cv::Point> cornors1;
    for (int i = 0; i < lines.size(); i++)
    {
        cv::Point cornor = computeIntersect(lines[i], lines[(i + 1) % lines.size()]);
        cornors4.push_back(cornor);
    }
    //绘制出四条边
    for (int i = 0; i < cornors4.size(); i++)
    {
        line(mat, cornors4[i], cornors4[(i + 1) % cornors4.size()], Scalar(0, 0, 255), 5);
    }


    return;
}

const int indent = 10;

Mat cutImage(Mat &src) {
	cv::Point2f srcVertex[4], dstVertex[4];
	srcVertex[0].x = cornors4[1].x ;
	srcVertex[0].y = cornors4[1].y ;
	srcVertex[1].x = cornors4[2].x ;
	srcVertex[1].y = cornors4[2].y ;
	srcVertex[2].x = cornors4[0].x ;
	srcVertex[2].y = cornors4[0].y ;
	srcVertex[3].x = cornors4[3].x ;
	srcVertex[3].y = cornors4[3].y ;
	/*
	Mat MoveImage(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));
	double angle = atan2( (srcVertex[1].y - srcVertex[0].y),(srcVertex[1].x - srcVertex[0].x));
	angle = angle * 180 / 3.1415;
	Mat M = getRotationMatrix2D(srcVertex[0], angle, 1);//计算旋转的仿射变换矩阵
	warpAffine(src, MoveImage, M, Size(src.cols, src.rows));//仿射变换
	imshow("rotate", MoveImage);
	*/
	dstVertex[0].x = cornors4[1].x;
	dstVertex[0].y = cornors4[1].y;
	dstVertex[1].x = cornors4[2].x;
	dstVertex[1].y = cornors4[1].y;
	dstVertex[2].x = cornors4[1].x;
	dstVertex[2].y = cornors4[0].y;
	dstVertex[3].x = cornors4[2].x;
	dstVertex[3].y = cornors4[0].y;

	Mat warp_mat = cv::getPerspectiveTransform(srcVertex, dstVertex);

	cv::Mat perspective;
	cv::warpPerspective(src, perspective, warp_mat, cv::Size(src.cols, src.rows), cv::INTER_LINEAR);


	//cvWarpPerspective(srcImg, dstImg, warp_mat, CV_WARP_FILL_OUTLIERS, cvScalarAll(255));
	Rect rectInImage;
	//rectInImage = cvRect(0,0 , img->width, img->height*0.5);
	int x0 = max(int(dstVertex[0].x - indent), 0);
	int y0 = max(int(dstVertex[0].y - indent), 0);
	int width = (srcVertex[1].x - srcVertex[0].x) + indent * 2;
	int height = (srcVertex[2].y - srcVertex[0].y) + indent * 2;
	if (x0 + width > src.cols)
		width = (srcVertex[1].x - srcVertex[0].x);
	if (y0 + height > src.rows)
		height = (srcVertex[2].y - srcVertex[0].y);
	rectInImage = Rect(x0,y0,width,height);//角点坐标
	Size size;
	size.width = rectInImage.width;
	size.height = rectInImage.height;

	Mat cutImg;// = Mat(size, perspective.type);
	Mat roi = perspective(rectInImage);
	roi.copyTo(cutImg);
	//cutImg = cvCreateImage(size, dstImg->depth, dstImg->nChannels);
	return cutImg;
}

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

Mat mycvPerspectiveCorrectionImage(Mat& image)
{
	Mat cannyMat;
	Canny(image, cannyMat, 60, 220, 3);

	CPerspective findVertex;
	findVertex.setMinVote(90);
	findVertex.setLineLengthAndGap(100, 80);
	findVertex.findLines(cannyMat);
	vector<Vec4i> horline = findVertex.drawVerticalLines(image);
	vector<Vec4i> verline = findVertex.drawHorizontalLines(image);

	cv::Point2f srcVertex[4], dstVertex[4];
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

	Mat warp_mat = cv::getPerspectiveTransform(srcVertex, dstVertex);

	cv::Mat perspective;
	cv::warpPerspective(image, perspective, warp_mat, cv::Size(image.cols, image.rows), cv::INTER_LINEAR);


	return perspective;
}


//轮廓按照面积大小升序排序
bool ascendSort(vector<Point> a, vector<Point> b) {
    return a.size() < b.size();

}

//轮廓按照面积大小降序排序
bool descendSort(vector<Point> a, vector<Point> b) {
    return a.size() > b.size();
}

Mat removeNoise(Mat& src) {
    Mat thresholdImage;
    //bitwise_not(grayImage, grayImage);
    threshold(src, thresholdImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    //imshow("2", thresholdImage);
    //Mat resultImage;
    //thresholdImage.copyTo(resultImage);
    vector< vector< Point> > contours;  //用于保存所有轮廓信息
    vector< vector< Point> > contours2; //用于保存面积不足100的轮廓
    vector<Point> tempV;				//暂存的轮廓

    findContours(thresholdImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    //cv::Mat labels;
    //int N = connectedComponents(resultImage, labels, 8, CV_16U);
    //findContours(labels, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


    //轮廓按照面积大小进行升序排序
    sort(contours.begin(), contours.end(), ascendSort);//升序排序
    vector<vector<Point> >::iterator itc = contours.begin();
    int i = 0;
    while (itc != contours.end())
    {
        //获得轮廓的矩形边界
        Rect rect = boundingRect(*itc);
        int x = rect.x;
        int y = rect.y;
        int w = rect.width;
        int h = rect.height;
        //绘制轮廓的矩形边界
        //cv::rectangle(srcImage, rect, { 0, 0, 255 }, 1);
        //保存图片
        char str[10];


        if (itc->size() < 200)
        {
            //把轮廓面积不足100的区域，放到容器contours2中，
            tempV.push_back(Point(x, y));
            tempV.push_back(Point(x, y + h));
            tempV.push_back(Point(x + w, y + h));
            tempV.push_back(Point(x + w, y));
            contours2.push_back(tempV);
            /*也可以直接用：contours2.push_back(*itc);代替上面的5条语句*/
            //contours2.push_back(*itc);

            //删除轮廓面积不足100的区域，即用白色填充轮廓面积不足100的区域：
            cv::drawContours(src, contours2, -1, Scalar(255, 255, 255), CV_FILLED);
        }

        tempV.clear();
        ++itc;
    }
    return src;
}

jobject mat_to_bitmap(JNIEnv * env, Mat & src, bool needPremultiplyAlpha, jobject bitmap_config){
    jclass java_bitmap_class = (jclass)env->FindClass("android/graphics/Bitmap");
    jmethodID mid = env->GetStaticMethodID(java_bitmap_class,
                                           "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");

    jobject bitmap = env->CallStaticObjectMethod(java_bitmap_class,
                                                 mid, src.size().width, src.size().height, bitmap_config);
    AndroidBitmapInfo  info;
    void*              pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888){
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(src.type() == CV_8UC1){
                cvtColor(src, tmp, CV_GRAY2RGBA);
            }else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, CV_RGB2RGBA);
            }else if(src.type() == CV_8UC4){
                if(needPremultiplyAlpha){
                    cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                }else{
                    src.copyTo(tmp);
                }
            }
        }else{
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if(src.type() == CV_8UC1){
                cvtColor(src, tmp, CV_GRAY2BGR565);
            }else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, CV_RGB2BGR565);
            }else if(src.type() == CV_8UC4){
                cvtColor(src, tmp, CV_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return bitmap;
    }catch(cv::Exception e){
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return bitmap;
    }catch (...){
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return bitmap;
    }
}

Mat bitmap_to_mat(JNIEnv *env,jobject bitmap){
    AndroidBitmapInfo bitmapInfo;
    uint32_t* storedBitmapPixels = NULL;
    int pixelsCount;
    int ret = -1;

    // 读取bitmap基本信息
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &bitmapInfo)) < 0) {
        //return NULL;
    }


    // 这里只处理RGBA_888类型的bitmap
    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        //return NULL;
    }

        // 提取像素值
    void* bitmapPixels = NULL;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels)) < 0) {
        //return NULL;
    }

    // 生成openCV Mat矩阵
    Mat srcMat(Size(bitmapInfo.width, bitmapInfo.height), CV_8UC4);
    pixelsCount = bitmapInfo.height * bitmapInfo.width;
    memcpy(srcMat.data, bitmapPixels, sizeof(uint32_t) * pixelsCount);
    AndroidBitmap_unlockPixels(env, bitmap);
    return srcMat;
}


extern "C" {

JNIEXPORT jobject JNICALL Java_com_example_pynixs_dodistortion_OpenCVUtils_getDistortion
        (JNIEnv *env, jobject thiz, jobject src, jobject frame) {

    Mat mSrc=bitmap_to_mat(env,src);
    Mat mFrame=bitmap_to_mat(env,frame);

    Mat fuse(mFrame.rows, mFrame.cols, CV_8UC1);
	cvtColor(mFrame, fuse, CV_BGR2GRAY);

    //Mat dst = Mat::zeros(mSrc.rows, mSrc.cols, CV_8UC1);
    //resize(fuse, dst, dst.size());

    vector<vector<Point>> contours;
    Mat Biny;
    threshold(fuse, Biny, 30, 200, CV_THRESH_BINARY);
    Mat conVt(Biny.size(), CV_8UC1);
    //cvtColor(Biny, conVt, CV_BGR2GRAY);
    Biny.convertTo(conVt, CV_8UC1);
    findContours(conVt, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    Mat result(conVt.size(), CV_8U, Scalar(255));
    drawContours(result, contours, -1, Scalar(0), 2);
    //removeNoise(result);  //去噪

    Mat proImg;
    cvtColor(result, proImg, CV_GRAY2RGB);
    resize(result, proImg, mSrc.size());
    processImage(proImg);

    Mat cutImg = cutImage(mSrc);

    //Mat outImg = mycvPerspectiveCorrectionImage(cutImg);

//get source bitmap's config
    jclass java_bitmap_class = (jclass)env->FindClass("android/graphics/Bitmap");
    jmethodID mid = env->GetMethodID(java_bitmap_class, "getConfig", "()Landroid/graphics/Bitmap$Config;");
    jobject bitmap_config = env->CallObjectMethod(src, mid);
    jobject _bitmap = mat_to_bitmap(env,cutImg,false,bitmap_config);

    return _bitmap;

}

}