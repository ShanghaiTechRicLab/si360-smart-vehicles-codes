#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  cv::Mat image = cv::imread(argv[1]);

  cv::imshow("Image", image);
  cv::waitKey(0);
}
