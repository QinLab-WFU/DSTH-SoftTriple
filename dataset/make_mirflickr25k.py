import os
import scipy.io as scio
import numpy as np

# mirflickr25k_annotations_v080 and mirflickr
# mkdir mat
# mv make_mirflickr25k.py mat
# python make_mirflickr25k.py

# 设置根目录
root_dir = "./flickr25k/"

# 设置存放注释文件的路径
file_path = os.path.join(root_dir, "mirflickr25k_annotations_v080")

# 获取注释文件列表，排除包含"_r1"和"README"的文件
file_list = os.listdir(file_path)
file_list = [item for item in file_list if "_r1" not in item and "README" not in item]

print("class num:", len(file_list))

# 为每个类别创建索引
class_index = {}
for i, item in enumerate(file_list):
    class_index.update({item: i})

# 创建标签字典
label_dict = {}
for path_id in file_list:
    path = os.path.join(file_path, path_id)
    with open(path, "r") as f:
        for item in f:
            item = item.strip()
            if item not in label_dict:
                label = np.zeros(len(file_list))
                label[class_index[path_id]] = 1
                label_dict.update({item: label})
            else:
                # print()
                label_dict[item][class_index[path_id]] = 1

# print(label_dict)
print("create label:", len(label_dict))
keys = list(label_dict.keys())
keys.sort()

# 生成标签列表
labels = []
for key in keys:
    labels.append(label_dict[key])

print("labels created:", len(labels))
labels = {"category": labels}

# 图片索引路径
# PATH = os.path.join(root_dir, "mirflickr25k", "mirflickr")
PATH = os.path.join(root_dir, "mirflickr")
index = [os.path.join(PATH, "im" + item + ".jpg") for item in keys]

print("index created:", len(index))
index = {"index": index}

# 提取标题信息
captions_path = os.path.join(root_dir, "mirflickr/meta/tags")
captions_list = os.listdir(captions_path)
captions_dict = {}
for item in captions_list:
    id_ = item.split(".")[0].replace("tags", "")
    caption = ""
    with open(os.path.join(captions_path, item), "r") as f:
        for word in f.readlines():
            caption += word.strip() + " "
    caption = caption.strip()
    captions_dict.update({id_: caption})

captions = []

for item in keys:
    captions.append([captions_dict[item]])

print("captions created:", len(captions))
captions = {"caption": captions}

# 保存为.mat文件
scio.savemat("./flickr25k/index.mat", index)
scio.savemat("./flickr25k/caption.mat", captions)
scio.savemat("./flickr25k/label.mat", labels)
