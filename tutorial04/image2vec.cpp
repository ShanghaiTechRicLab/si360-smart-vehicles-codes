#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

#include "DBoW3/DBoW3.h"
#include "DBoW3/DescManip.h"

void PrintHelp(char* argv[]) {
  std::cout << "Usage: " << argv[0]
            << " [database path]  [image path] [feature type]" << std::endl;
  std::cout << std::endl;

  std::cout << "\t[feature type]" << std::endl;
  std::cout << "\t\t"
            << "orb" << std::endl;
  std::cout << "\t\t"
            << "brisk" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    PrintHelp(argv);
    std::exit(0);
  }

  std::string database_path = argv[1];
  std::string image_path = argv[2];
  std::string feature_type = argv[3];

  // read database
  std::cout << "reading database: " << database_path << std::endl;

  DBoW3::Vocabulary vocab(database_path);
  if (vocab.empty()) {
    std::cerr << "the vocabulary is empty" << std::endl;
    std::exit(1);
  }

  // read image and extract features
  std::cout << "reading image: " << image_path << std::endl;
  cv::Mat image = cv::imread(image_path, 0);
  if (image.empty()) {
    std::cerr << "the image is empty" << std::endl;
    std::exit(1);
  }
  std::cout << "extracting features" << std::endl;
  cv::Ptr<cv::Feature2D> detector;
  if (feature_type == "orb") {
    detector = cv::ORB::create();
  } else if (feature_type == "brisk") {
    detector = cv::BRISK::create();
  } else {
    std::cerr << "unknown feature type: " << feature_type << std::endl;
    std::exit(1);
  }

  // transform features to vector
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptor;
  detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

  DBoW3::BowVector bow_vector;
  vocab.transform(descriptor, bow_vector);
  std::cout << "bow vector: " << bow_vector << std::endl;
}