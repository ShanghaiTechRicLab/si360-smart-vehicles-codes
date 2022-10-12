#include <iostream>

#include <pcl/features/3dsc.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/vfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>

#include "simple_timer.hpp"

void PrintUsage(const char *progname) {
  std::cout << "Usage: " << progname << " [input.pcd]" << std::endl;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    PrintUsage(argv[0]);
    std::exit(0);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

  // 1. read point cloud
  pcl::io::loadPCDFile(argv[1], *cloud);
  std::cout << "Loaded " << cloud->width * cloud->height << " data points from "
            << argv[1] << std::endl;

  // 2. compute normals
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setRadiusSearch(0.03);
  normal_estimator.setInputCloud(cloud);
  StopWatchTimer normal_estimate_timer{"normal_estimate_timer"};
  normal_estimate_timer.start();
  normal_estimator.compute(*normals);
  normal_estimate_timer.stop();
  std::cout << "Normal estimate time: "
            << normal_estimate_timer.elapsed_time_ms() << " ms" << std::endl;

  // 3. compute pfh features
  // Create the PFH estimation class, and pass the input dataset+normals to it
  pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
  pfh.setInputCloud(cloud);
  pfh.setInputNormals(normals);
  // Create an empty kdtree representation, and pass it to the PFH estimation
  // object. Its content will be filled inside the object, based on the given
  // input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(
      new pcl::search::KdTree<pcl::PointXYZ>());
  // pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new
  // pcl::KdTreeFLANN<pcl::PointXYZ> ()); -- older call for PCL 1.5-
  pfh.setSearchMethod(tree2);
  // Output datasets
  pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs(
      new pcl::PointCloud<pcl::PFHSignature125>());
  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to
  // estimate the surface normals!!!
  pfh.setRadiusSearch(0.05);
  // Compute the features
  StopWatchTimer pfh_timer{"pfh"};
  pfh_timer.start();
  pfh.compute(*pfhs);
  pfh_timer.stop();
  // pfhs->size () should have the same size as the input cloud->size ()*
  std::cout << "Detect " << pfhs->size() << " PFH featrues using "
            << pfh_timer.elapsed_time_ms() << "ms" << std::endl;

  // 4. compute fpfh features
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud(cloud);
  fpfh.setInputNormals(normals);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree3(
      new pcl::search::KdTree<pcl::PointXYZ>());
  fpfh.setSearchMethod(tree3);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(
      new pcl::PointCloud<pcl::FPFHSignature33>());
  fpfh.setRadiusSearch(0.05);
  StopWatchTimer fpfh_timer{"fpfh"};
  fpfh_timer.start();
  fpfh.compute(*fpfhs);
  fpfh_timer.stop();
  std::cout << "Detect " << fpfhs->size() << " FPFH featrues using "
            << fpfh_timer.elapsed_time_ms() << "ms" << std::endl;

  pcl::visualization::PCLHistogramVisualizer vis;
  vis.addFeatureHistogram<pcl::PFHSignature125>(*pfhs, pfhs->size(), "PFH");
  vis.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs, fpfhs->size(), "FPFH");
  vis.spin();

  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(
      new pcl::PointCloud<pcl::VFHSignature308>());
  vfh.setInputCloud(cloud);
  vfh.setInputNormals(normals);
  vfh.setSearchMethod(tree);
  vfh.setNormalizeBins(true);
  vfh.setNormalizeDistance(false);
  StopWatchTimer vfh_timer{"vfh"};
  vfh_timer.start();
  vfh.compute(*vfhs);
  vfh_timer.stop();
  std::cout << "Detect " << vfhs->size() << " VFH featrues using "
            << vfh_timer.elapsed_time_ms() << " ms" << std::endl;
  std::cout << "the first point's vfh descriptor is " << std::endl;
  std::cout << (*vfhs)[0] << std::endl;

  // 5.compute the SHOT features
  pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
  shot.setInputCloud(cloud);
  shot.setInputNormals(normals);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree4(
      new pcl::search::KdTree<pcl::PointXYZ>());
  shot.setSearchMethod(tree4);
  pcl::PointCloud<pcl::SHOT352>::Ptr shots(new pcl::PointCloud<pcl::SHOT352>());
  shot.setRadiusSearch(0.05);
  StopWatchTimer shot_timer{"shot"};
  shot_timer.start();
  shot.compute(*shots);
  shot_timer.stop();
  std::cout << "Detect " << shots->size() << " SHOT featrues using "
            << shot_timer.elapsed_time_ms() << " ms " << std::endl;
  std::cout << "the first point's shot descriptor is " << std::endl;
  std::cout << (*shots)[0] << std::endl;

  // 6. compute for 3D shape context
  pcl::ShapeContext3DEstimation<pcl::PointXYZ, pcl::Normal,
                                pcl::ShapeContext1980>
      sc3d;
  sc3d.setInputCloud(cloud);
  sc3d.setInputNormals(normals);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree5(
      new pcl::search::KdTree<pcl::PointXYZ>());
  sc3d.setSearchMethod(tree5);
  pcl::PointCloud<pcl::ShapeContext1980>::Ptr sc3ds(
      new pcl::PointCloud<pcl::ShapeContext1980>());
  sc3d.setRadiusSearch(0.05);
  sc3d.setMinimalRadius(0.05 / 10.0);
  sc3d.setPointDensityRadius(0.05 / 5.0);
  StopWatchTimer sc3d_timer{"sc3d"};
  sc3d_timer.start();
  sc3d.compute(*sc3ds);
  sc3d_timer.stop();
  std::cout << "Detect " << sc3ds->size() << " 3D shape context featrues using "
            << sc3d_timer.elapsed_time_ms() << " ms" << std::endl;
  std::cout << "the first point's shape context descriptor is " << std::endl;
  std::cout << (*sc3ds)[0] << std::endl;
}
