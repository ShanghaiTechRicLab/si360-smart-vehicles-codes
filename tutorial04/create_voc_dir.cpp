#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3/DBoW3.h"
#include "DBoW3/DescManip.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>

#include "simple_timer.hpp"

namespace fs = boost::filesystem;

std::vector<fs::path> GetAllImagePaths(fs::path const &root) {
  std::vector<fs::path> image_paths;
  if (fs::exists(root) && fs::is_directory(root)) {
    for (auto const &entry : fs::recursive_directory_iterator(root)) {
      if (fs::is_regular_file(entry) && (entry.path().extension() == ".png" ||
                                         entry.path().extension() == ".jpg" ||
                                         entry.path().extension() == ".jpeg")) {
        image_paths.push_back(entry.path());
      }
    }
  }
  return image_paths;
}

std::vector<cv::Mat> loadFeatures(std::vector<fs::path> path_to_images,
                                  std::string descriptor = "") {
  // select detector
  cv::Ptr<cv::Feature2D> fdetector;
  if (descriptor == "orb")
    fdetector = cv::ORB::create();
  else if (descriptor == "brisk")
    fdetector = cv::BRISK::create();
  else
    throw std::runtime_error("Invalid descriptor");
  assert(!descriptor.empty());
  std::vector<cv::Mat> features;

  std::cout << "Extracting   features..." << std::endl;
  for (size_t i = 0; i < path_to_images.size(); ++i) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::cout << "reading image: " << path_to_images[i] << std::endl;
    cv::Mat image = cv::imread(path_to_images[i].string().c_str(), 0);
    if (image.empty())
      throw std::runtime_error("Could not open image" +
                               path_to_images[i].string());
    std::cout << "extracting features" << std::endl;
    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    features.push_back(descriptors);
    std::cout << "done detecting features" << std::endl;
  }
  return features;
}

void PrintUsage() {
  std::cout << "Usage: ./create_voc_dir <feature_name> <image_dir> <voc_dir> "
               "<voc_name>"
            << std::endl;
  std::cout << "  featurename is:" << std::endl;
  std::cout << "     orb" << std::endl;
  std::cout << "     brisk" << std::endl;
  std::exit(0);
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    PrintUsage();
  }

  std::string feature_name = argv[1];
  std::string image_dir = argv[2];
  std::string voc_dir = argv[3];
  std::string voc_name = argv[4];

  std::cout << "Feature name: " << feature_name << std::endl;
  std::cout << "Image dir: " << image_dir << std::endl;
  std::cout << "Voc dir: " << voc_dir << std::endl;
  std::cout << "Voc name: " << voc_name << std::endl;

  std::vector<fs::path> image_paths = GetAllImagePaths(image_dir);
  std::cout << "Number of images: " << image_paths.size() << std::endl;

  StopWatchTimer feat_timer{"load featrues"};
  feat_timer.start();
  auto features = loadFeatures(image_paths, feature_name);
  feat_timer.stop();

  const int k = 9;
  const int L = 3;
  const DBoW3::WeightingType weight = DBoW3::TF_IDF;
  const DBoW3::ScoringType score = DBoW3::L1_NORM;
  DBoW3::Vocabulary voc(k, L, weight, score);

  std::cout << "Creating a small " << k << "^" << L << " vocabulary..."
            << std::endl;
  StopWatchTimer voc_timer{"create voc"};
  voc_timer.start();
  voc.create(features);
  voc_timer.stop();
  // create the directory if not exists
  fs::path voc_dir_path = fs::path(voc_dir);
  if (!fs::exists(voc_dir_path)) {
    fs::create_directory(voc_dir_path);
  }
  fs::path voc_path = voc_dir_path / fs::path(voc_name);
  std::cerr << "Saving " << voc_path << std::endl;
  StopWatchTimer save_timer{"save voc"};
  save_timer.start();
  voc.save(voc_path.string());
  save_timer.stop();


  feat_timer.print_stats();
  voc_timer.print_stats();
  save_timer.print_stats();
}
