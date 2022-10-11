#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

int main(int argc, char* argv[]) {
  cv::Mat image = cv::imread(argv[1], 0);

  cv::imshow("Origin Image", image);
  cv::waitKey(0);


  cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
  std::vector<cv::KeyPoint> sift_keypoints;
  siftPtr->detect(image, sift_keypoints);
  std::cout << "SIFT feature detector detect " << sift_keypoints.size() << " points" << std::endl;
  cv::Mat output;
  cv::drawKeypoints(image, sift_keypoints, output);
  cv::imshow("Image with SIFT feature points", output);
  cv::waitKey(0);

  cv::Ptr<cv::xfeatures2d::SURF> surfPtr = cv::xfeatures2d::SURF::create();
  std::vector<cv::KeyPoint> surf_keypoints;
  surfPtr->detect(image, surf_keypoints);
  std::cout << "surf feature detector detect " << surf_keypoints.size() << " points" << std::endl;
  cv::Mat surf_output;
  cv::drawKeypoints(image, surf_keypoints, surf_output);
  cv::imshow("Image with surf feature points", surf_output);
  cv::waitKey(0);

  const int block_size = 2;
  const int aperture_size = 3;
  const double k = 0.04;
  const int thresh = 200;
  std::vector<cv::KeyPoint> harris_keypoints;
  cv::Mat harris_dst = cv::Mat::zeros(image.size(), CV_32FC1);
  cv::cornerHarris(image, harris_dst, block_size, aperture_size, k); 
  cv::Mat harris_dst_norm, harris_dst_norm_scaled;
  cv::normalize(harris_dst, harris_dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(harris_dst_norm, harris_dst_norm_scaled);
  for(int i = 0; i < harris_dst_norm.rows; i++){
    for(int j = 0; j < harris_dst_norm.cols; j++){
      if(static_cast<int>(harris_dst_norm.at<float>(i, j)) > thresh)
      {
         cv::circle(harris_dst_norm_scaled, cv::Point(j, i), 5, cv::Scalar(0), 2, 8, 0);
         harris_keypoints.emplace_back(cv::Point(j, i), 5);
      }
    }
  }
  std::cout << "harris 2d detect " << harris_keypoints.size() << " corner points" << std::endl;
  cv::imshow("Image with harris2d corner point", harris_dst_norm_scaled);
  cv::waitKey(0);

  // insert 


  return 0;
}
