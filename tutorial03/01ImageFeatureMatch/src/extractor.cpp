#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "simple_timer.hpp"

int main(int argc, char *argv[]) {
  // read image as grayscale
  cv::Mat image = cv::imread(argv[1], 0);

  // display read image
  cv::imshow("Origin Image", image);
  cv::waitKey(0);

  // 1. sift feature extractor
  StopWatchTimer sift_timer{"sift_timer"};
  cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
  std::vector<cv::KeyPoint> sift_keypoints;
  sift_timer.start();
  siftPtr->detect(image, sift_keypoints);
  sift_timer.stop();
  std::cout << "SIFT feature detector detect " << sift_keypoints.size()
            << " points" << std::endl;
  sift_timer.print_stats();
  cv::Mat output;
  cv::drawKeypoints(image, sift_keypoints, output);
  cv::imshow("Image with SIFT feature points", output);
  cv::waitKey(0);

  // 2. surf featrue extractor
  StopWatchTimer surf_timer{"sift_timer"};
  cv::Ptr<cv::xfeatures2d::SURF> surfPtr = cv::xfeatures2d::SURF::create();
  std::vector<cv::KeyPoint> surf_keypoints;
  surf_timer.start();
  surfPtr->detect(image, surf_keypoints);
  surf_timer.stop();
  std::cout << "surf feature detector detect " << surf_keypoints.size()
            << " points" << std::endl;
  surf_timer.print_stats();
  cv::Mat surf_output;
  cv::drawKeypoints(image, surf_keypoints, surf_output);
  cv::imshow("Image with surf feature points", surf_output);
  cv::waitKey(0);

  // 3. harries corner extractor
  const int block_size = 2;
  const int aperture_size = 3;
  const double k = 0.04;
  const int thresh = 200;
  StopWatchTimer harris_timer{"harris_timer"};
  std::vector<cv::KeyPoint> harris_keypoints;
  cv::Mat harris_dst = cv::Mat::zeros(image.size(), CV_32FC1);
  harris_timer.start();
  cv::cornerHarris(image, harris_dst, block_size, aperture_size, k);
  harris_timer.stop();
  cv::Mat harris_dst_norm, harris_dst_norm_scaled;
  cv::normalize(harris_dst, harris_dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1,
                cv::Mat());
  cv::convertScaleAbs(harris_dst_norm, harris_dst_norm_scaled);
  for (int i = 0; i < harris_dst_norm.rows; i++) {
    for (int j = 0; j < harris_dst_norm.cols; j++) {
      if (static_cast<int>(harris_dst_norm.at<float>(i, j)) > thresh) {
        harris_keypoints.emplace_back(cv::Point(j, i), 5);
      }
    }
  }
  cv::Mat harris_output;
  cv::drawKeypoints(image, harris_keypoints, harris_output);
  std::cout << "harris 2d detect " << harris_keypoints.size()
            << " corner points" << std::endl;
  harris_timer.print_stats();
  cv::imshow("Image with harris2d corner point", harris_output);
  cv::waitKey(0);

  // 4. FAST feature detector
  StopWatchTimer fast_timer{"fast_timer"};
  cv::Ptr<cv::FeatureDetector> fast_detector = cv::ORB::create();
  std::vector<cv::KeyPoint> fast_keypoints;
  fast_timer.start();
  fast_detector->detect(image, fast_keypoints);
  fast_timer.stop();
  cv::Mat fast_output;
  cv::drawKeypoints(image, fast_keypoints, fast_output);
  std::cout << "fast detector dectect " << fast_keypoints.size()
            << " feature points" << std::endl;
  fast_timer.print_stats();
  cv::imshow("Image with fast feature points", fast_output);
  cv::waitKey(0);

  return 0;
}
