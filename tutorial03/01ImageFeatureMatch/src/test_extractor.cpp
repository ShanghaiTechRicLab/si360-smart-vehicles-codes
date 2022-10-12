#include "feature_extractor.h"

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage:" << std::endl;
    std::cout << "\t" << argv[0] << " [image] [extractor id]" << std::endl;
    std::cout << "\tExtractor ID:" << std::endl;
    std::cout << "\t SIFT -> 0" << std::endl;
    std::cout << "\t SURF -> 1" << std::endl;
    std::cout << "\t FAST -> 2" << std::endl;
    std::exit(0);
  }
  // read image as grayscale
  cv::Mat image = cv::imread(argv[1], 0);

  // display read image
  cv::imshow("Origin Image", image);
  cv::waitKey(0);

  FeatureExtractor *extractor =
      create_extractor(static_cast<ExtractorType>(std::atoi(argv[2])));

  std::vector<cv::KeyPoint> keypoints;
  extractor->detect(image, keypoints);
  std::cout << "extractor[" << extractor->get_name() << "] detect "
            << keypoints.size() << " points using "
            << extractor->get_detect_time() << " ms" << std::endl;
  cv::Mat output;
  cv::drawKeypoints(image, keypoints, output);
  cv::imshow("Image with feature points", output);
  cv::waitKey(0);

  return 0;
}

