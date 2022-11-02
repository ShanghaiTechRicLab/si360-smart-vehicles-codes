#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

#include "DBoW3/DBoW3.h"
#include "DBoW3/DescManip.h"

void PrintHelp(char *argv[]) {
  std::cout << "Usage: " << argv[0]
            << " [database path]  [image 1 path] [image 2 path] [feature type]"
            << std::endl;
  std::cout << std::endl;

  std::cout << "\t[feature type]" << std::endl;
  std::cout << "\t\t"
            << "orb" << std::endl;
  std::cout << "\t\t"
            << "brisk" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    PrintHelp(argv);
    std::exit(0);
  }

  std::string database_path = argv[1];
  std::string image1_path = argv[2];
  std::string image2_path = argv[3];
  std::string feature_type = argv[4];

  // read database
  std::cout << "reading database: " << database_path << std::endl;

  DBoW3::Vocabulary vocab(database_path);
  if (vocab.empty()) {
    std::cerr << "the vocabulary is empty" << std::endl;
    std::exit(1);
  }

  // read image and extract features
  std::cout << "reading image: " << image1_path << std::endl;
  cv::Mat image1 = cv::imread(image1_path, 0);
  if (image1.empty()) {
    std::cerr << "the image is empty" << std::endl;
    std::exit(1);
  }
  std::cout << "reading image: " << image2_path << std::endl;
  cv::Mat image2 = cv::imread(image2_path, 0);
  if (image2.empty()) {
    std::cerr << "the image is empty" << std::endl;
    std::exit(1);
  }
  std::cout << "extracting features" << std::endl;
  cv::Ptr<cv::Feature2D> detector1;
  cv::Ptr<cv::Feature2D> detector2;
  if (feature_type == "orb") {
    detector1 = cv::ORB::create();
    detector2 = cv::ORB::create();
  } else if (feature_type == "brisk") {
    detector1 = cv::BRISK::create();
    detector2 = cv::BRISK::create();
  } else {
    std::cerr << "unknown feature type: " << feature_type << std::endl;
    std::exit(1);
  }

  // transform features to vector
  std::vector<cv::KeyPoint> keypoints1;
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptor1;
  cv::Mat descriptor2;
  detector1->detectAndCompute(image1, cv::Mat(), keypoints1, descriptor1);
  detector2->detectAndCompute(image2, cv::Mat(), keypoints2, descriptor2);

  DBoW3::BowVector bow_vector1;
  DBoW3::BowVector bow_vector2;
  vocab.transform(descriptor1, bow_vector1);
  vocab.transform(descriptor2, bow_vector2);
  double score = vocab.score(bow_vector1, bow_vector2);
  std::cout << "similarity score: " << score << std::endl;
  return 0;
}
