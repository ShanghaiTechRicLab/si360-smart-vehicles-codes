# Use DBoW3 to detect loop clousure

## Prepare

1. Install `DBoW3`: follow [DBoW3](https://github.com/rmsalinas/DBow3)
2. Install `Boost`: `sudo apt install libboost-dev`
3. Install `cmake`

## Build

```bash
make build
cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
make -j $(nproc)
```

## Run

以 `tutorial04` 为基本目录

### `create_voc_dir` 从一个包含图片的文件夹中建立词典

```bash
Usage: ./create_voc_dir <feature_name> <image_dir> <voc_dir> <voc_name>
  featurename is:
     orb
     brisk
```

参数说明:

- `feautre_name`: 使用的特征提取方法，有 `orb` 和 `brisk` 两种
- `image_dir`: 用于建立词典的图片目录
- `voc_dir`: 用于保存建立的词典的目录
- `voc_name`: 建立的词典的名称

一个例子:

```bash
 ./build/create_voc_dir orb /data/2011_10_03/2011_10_03_drive_0027_sync/image_00 kitti 2011_10_03_drive_0027_sync_image_00.db
```

附件提供了建立好的测试词典: `kitti/2011_10_03_drive_0027_sync_image_00.db`

### `image2vec` 获得图片在词典中的单词描述向量

```bash
Usage: ./build/release/image2vec [voc path]  [image path] [feature type]

        [feature type]
                orb
                brisk
```

参数说明:

- `voc path`: 建立好的词典的路径
- `image path`: 需要转换的图片的路径
- `feature type`: 用于提取特征的方法： `orb` 或者 `brisk`

一个例子:

```bash
 ./build/image2vec kitti/2011_10_03_drive_0027_sync_image_00.db /data/2011_10_03/2011_10_03_drive_0027_sync/image_00/data/0000000012.png orb
```

### `image_similarity_score` 利用单词描述向量比较两张图的相似性

```bash
Usage: ./build/release/image_similarity_score [voc path]  [image 1 path] [image 2 path] [feature type]

        [feature type]
                orb
                brisk

```

参数说明:

- `voc path`: 建立好的词典的路径
- `image1 path`: 需要比较的图片 1 的路径
- `image2 path`: 需要比较的图片 2 的路径
- `feature type`: 用于提取特征的方法： `orb` 或者 `brisk`

一个例子:

```bash
./build/image_similarity_score kitti/2011_10_03_drive_0027_sync_image_00.db  \\
    /data/2011_10_03/2011_10_03_drive_0027_sync/image_00/data/0000000012.png \\
    /data/2011_10_03/2011_10_03_drive_0027_sync/image_00/data/0000000021.png \\
    orb
```

### `query_database` 建立数据库来查询相似图片

```bash
Usage: ./build/query_database <voc_path>  <image_to_create_database_dir> <query_image> <feature_name>
  feature_name is:
     orb
     brisk
```

参数说明:

- `voc path`: 建立好的词典的路径
- `image_to_create_database_dir`: 包含用来建立数据库的图片的目录，即包含目标图片的目录
- `query_image`: 需要查询相似图片的图片
- `feature type`: 用于提取特征的方法： `orb` 或者 `brisk`

一个例子:

```bash
./build/query_database kitti/2011_10_03_drive_0027_sync_image_00.db sample_image sample_image/0000000210.png orb
```
