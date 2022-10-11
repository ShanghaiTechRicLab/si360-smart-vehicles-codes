#  Image Feture Extracting And Matching Using OpenCV

## Prepare

You need to download the prebuilt OpenCV 4.6.0 package to your local directory. Look for Tutorial3-1.pdf for more information. (You can build by you self, see Below)

## Build the example

You need install `cmake`

```
# configure the build, change Release to Debug if you want to debug
# replace /your/path/to/opencv_install to your extracted path
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -D OpenCV_DIR=/your/path/to/opencv_install/lib/cmake/opencv4
cmake --build build -j $(nproc)
```

## Run the example

Here is three excutable file after build, we foucs on `1` and `3`:
1. `image_feature_extractor` show different detected feature points for one same image
2. `test_extractor_lib` a test file for `feature_extractor_lib`
3. `image_feature_matcher` show feature matching of two image with different features extracting method and different match method

For `image_feature_extractor`

```
./build/image_feature_extractor [image]

# for example

./build/image_feature_extractor kitti_00.png
```

For `image_feature_matcher`

```
Usage:
        ./builde/image_feature_matcher [image1] [image2] [extractor id] [matcher id]
        Extractor ID:
                 SIFT -> 0
                 SURF -> 1
                 FAST -> 2
         Matcher ID:
                 BruteForce-L2 -> 2
                 BruteForce-L1 -> 3
                 BruteForce-Hamming -> 4
                 BruteForce-HammingLUT -> 5
                 BruteForce-SL2 -> 6
```

```
# for example, use FAST and BruteForce-Hamming:
./builde/image_feature_matcher kitti_01.png kitti_03.png 2 4
```


### Optional: Build OpenCV by yourself

Download [OpenCV 4.6 soruce code](https://github.com/opencv/opencv/archive/refs/tags/4.6.0.tar.gz) and [OpenCV Contrib 4.6 soruce code](https://github.com/opencv/opencv_contrib/archive/refs/tags/4.6.0.tar.gz)

```
mkdir opencv && cd opencv
mkdir opencv_install
wget -O opencv_4.6.0.tar.gz https://github.com/opencv/opencv/archive/refs/tags/4.6.0.tar.gz
wget -O opencv_contrib_4.6.0.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/4.6.0.tar.gz
tar xvf opencv_4.6.0.tar.gz
tar xvf opencv_contrib_4.6.0.tar.gz
cd opencv-4.6.0
sudo apt install libgtk2.0-dev
# configure opencv to build with contrib and nonfree package and build with gui support
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=/home/dengqi/Source/app/opencv/opencv_install -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON -DOPENCV_ENABLE_NONFREE=ON
# build
cmake --build build -j $(nproc)
# install to opencv/opencv_install
cmake --install build
```