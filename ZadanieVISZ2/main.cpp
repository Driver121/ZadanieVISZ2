
#include <stdio.h>
#include <string>
#include <map>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "hough.h"

using namespace cv;

int  threshold = 2;

std::string img_path = "../Data/house.jpg";


const char* CW_IMG_ORIGINAL = "Result";
const char* CW_IMG_EDGE = "Canny Edge Detection";
const char* CW_ACCUMULATOR = "Accumulator";
const char* CW_IMG_BLUR = "Blur";

void doTransform(std::string, int threshold);


void usage(char * s)
{

	fprintf(stderr, "\n");
	fprintf(stderr, "%s -s <source file> [-t <threshold>] - hough transform. build: %s-%s \n", s, __DATE__, __TIME__);
	fprintf(stderr, "   s: path image file\n");
	fprintf(stderr, "   t: hough threshold\n");
	fprintf(stderr, "\nexample:  ./hough -s ./img/russell-crowe-robin-hood-arrow.jpg -t 195\n");
	fprintf(stderr, "\n");
}


// V funkcii main si vytvoríme okná do ktorých zobrazíme výsledky ,
// nastavíme ich velkos a ich polohu na obrazovke.
// V maine ešte zavoláme funkciu doTransform ktorá vykoná houghovú 
// tranosformáciu a zobrazí výsledky do okien ktoré sme si vytvorili 
int main(int argc, char** argv) {


	cv::namedWindow(CW_IMG_ORIGINAL, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(CW_IMG_BLUR, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(CW_IMG_EDGE, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(CW_ACCUMULATOR, cv::WINDOW_AUTOSIZE);

	cvMoveWindow(CW_IMG_ORIGINAL, 10, 10);
	cvMoveWindow(CW_IMG_EDGE, 680, 10);
	cvMoveWindow(CW_ACCUMULATOR, 680, 10);
	cvMoveWindow(CW_IMG_BLUR, 20, 10);

	doTransform(img_path, 0); // tu zadáme treshold 


	return 0;
}


// V tejto funkcii urobíme naèitánie obrázka, potom tento obrázok prejdeme pomocou filtrov a to filtrom 
//rozmazania , potom filtrom na dekeciu hrán ktorý tam obrázok upraví aj do èierno bielej
// potom pošleme takéto spracovaný obraz do našej funkcie v ktoréj detekujeme èiary tam si
//naplníme akumulátor a potom v tomto akumulátore hladáme najlepšie možnosti 
// Nasledne najlepšie výsledky zobrazíme 
void doTransform(std::string file_path, int threshold)
{

	cv::Mat img_edge;
	cv::Mat img_blur;

	cv::Mat img_ori = cv::imread(file_path, 1);
	cv::blur(img_ori, img_blur, cv::Size(5, 5));
	cv::Canny(img_blur, img_edge, 100, 150, 3);

	int w = img_edge.cols;
	int h = img_edge.rows;

	//Transfromácia 
	zad::Hough hough;
	hough.Transform(img_edge.data, w, h);



	if (threshold == 0)
		threshold = w>h ? w / 4 : h / 4;

	while (1)
	{
		cv::Mat img_res = img_ori.clone();

		//Hladám v akumulátore 
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines = hough.GetLines(threshold);

		//Vykreslíme výsledky
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;
		for (it = lines.begin(); it != lines.end(); it++)
		{
			cv::line(img_res, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0, 0, 255), 2, 8);
		}

		//Zobrazím všetky 
		int aw, ah, maxa;
		aw = ah = maxa = 0;
		const unsigned int* accu = hough.GetAccu(&aw, &ah);

		for (int p = 0; p<(ah*aw); p++)
		{
			if ((int)accu[p] > maxa)
				maxa = accu[p];
		}
		double contrast = 1.0;
		double coef = 255.0 / (double)maxa * contrast;

		cv::Mat img_accu(ah, aw, CV_8UC3);
		for (int p = 0; p<(ah*aw); p++)
		{
			unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
			img_accu.data[(p * 3) + 0] = 255;
			img_accu.data[(p * 3) + 1] = 255 - c;
			img_accu.data[(p * 3) + 2] = 255 - c;
		}

		cv::imshow(CW_IMG_BLUR, img_blur);
		cv::imshow(CW_IMG_ORIGINAL, img_res);
		cv::imshow(CW_IMG_EDGE, img_edge);
		cv::imshow(CW_ACCUMULATOR, img_accu);

		char c = cv::waitKey(360000);
		if (c == '+')
			threshold += 5;
		if (c == '-')
			threshold -= 5;
		if (c == 27)
			break;
	}
}