#include "feature_extractor.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "simple_timer.hpp"

void PrintUsage(const char *progname) {
  std::cout << "Usage:" << std::endl;
  std::cout << "\t" << progname
            << " [image1] [image2] [extractor id] [matcher id]" << std::endl;
  std::cout << "\tExtractor ID:" << std::endl;
  std::cout << "\t\t SIFT -> 0" << std::endl;
  std::cout << "\t\t SURF -> 1" << std::endl;
  std::cout << "\t\t FAST -> 2" << std::endl;
  std::cout << "\t Matcher ID:" << std::endl;
  std::cout << "\t\t BruteForce-L2 -> 2" << std::endl;
  std::cout << "\t\t BruteForce-L1 -> 3" << std::endl;
  std::cout << "\t\t BruteForce-Hamming -> 4" << std::endl;
  std::cout << "\t\t BruteForce-HammingLUT -> 5" << std::endl;
  std::cout << "\t\t BruteForce-SL2 -> 6" << std::endl;
}

int main(int argc, char *argv[]) {
  // display help messsage
  if (argc < 5) {
    PrintUsage(argv[0]);
    std::exit(0);
  }

  if (std::atoi(argv[3]) > 2 || std::atoi(argv[3]) < 0) {
    PrintUsage(argv[0]);
    std::exit(0);
  }

  if (std::atoi(argv[4]) > 6 || std::atoi(argv[4]) < 2) {
    PrintUsage(argv[0]);
    std::exit(0);
  }

  // 1. read imag in grayscale
  cv::Mat input1 = cv::imread(argv[1], 0);
  cv::Mat input2 = cv::imread(argv[2], 0);

  // 2. extract feature points from two image
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  FeatureExtractor *extractor =
      create_extractor(static_cast<ExtractorType>(std::atoi(argv[3])));
  extractor->detect(input1, keypoints1);
  std::cout << "Extractor " << extractor->get_name() << " use "
            << extractor->get_detect_time() << " ms to detect "
            << keypoints1.size() << " points from input image 1" << std::endl;
  extractor->detect(input2, keypoints2);
  std::cout << "Extractor " << extractor->get_name() << " use "
            << extractor->get_detect_time() << " ms to detect "
            << keypoints2.size() << " points from input image 2" << std::endl;

  // 3. use BRIEF descriptor to descripe the keypoints
  cv::Mat descriptors1, descriptors2;
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  StopWatchTimer descriptor_timer1{"descriptor_timer1"};
  StopWatchTimer descriptor_timer2{"descriptor_tiemr2"};
  descriptor_timer1.start();
  descriptor->compute(input1, keypoints1, descriptors1);
  descriptor_timer1.stop();
  descriptor_timer2.start();
  descriptor->compute(input2, keypoints2, descriptors2);
  descriptor_timer2.stop();
  descriptor_timer1.print_stats();
  descriptor_timer2.print_stats();

  // 4. match the two image use brutforce hamming distance
  std::vector<cv::DMatch> matches;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
      static_cast<cv::DescriptorMatcher::MatcherType>(atoi(argv[4])));
  StopWatchTimer match_timer("match timer");
  matcher->match(descriptors1, descriptors2, matches);
  std::cout << "Descriptor Matcher" << atoi(argv[4]) << "use "
            << match_timer.elapsed_time_ms() << " ms to find " << matches.size()
            << " matches" << std::endl;

  // 5. remove the bad matches
  double min_dist = 10000, max_dist = 0;
  for (std::size_t i = 0; i < descriptors1.rows; ++i) {
    double dist = matches[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }
  std::cout << "Max distance: " << max_dist << std::endl;
  std::cout << "Min distance: " << min_dist << std::endl;
  // if the distance is larger than 2 times of min distance, then it is a bad
  // we also have to make sure the distance is larger than 30 (empirically
  // value)
  std::vector<cv::DMatch> good_matches;
  for (std::size_t i = 0; i < descriptors1.rows; ++i) {
    if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  std::cout << "Good matches: " << good_matches.size() << std::endl;

  // 6. draw the matches
  cv::Mat img_matches, img_good_matches;
  cv::drawMatches(input1, keypoints1, input2, keypoints2, matches, img_matches);
  cv::drawMatches(input1, keypoints1, input2, keypoints2, good_matches,
                  img_good_matches);
  cv::imshow("All matches", img_matches);
  cv::imshow("Good matches", img_good_matches);
  cv::waitKey(0);

  return 0;
}

