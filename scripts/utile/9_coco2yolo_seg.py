import os
import json
import shutil

def write_yolo_txt_file(txt_file_path,label_seg_x_y_list):
    if not os.path.exists(txt_file_path):
        with open(txt_file_path, "w") as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')
    else:
        with open(txt_file_path, "a") as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')

def read_json(in_json_path,img_dir,target_dir):
    with open(in_json_path, "r", encoding='utf-8') as f:
        # json.load数据到变量json_data
        json_data = json.load(f) 

    # print(len(json_data['annotations']))
    # print(len(json_data['images']))
    # print(len(json_data['categories']))

    for annotation in json_data['annotations']: # 遍历标注数据信息
        # print(annotation)
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        for image in json_data['images']: # 遍历图片相关信息
            if image['id'] == image_id:
                width = image['width'] # 图片宽
                height = image['height'] # 图片高
                img_file_name = image['file_name'] # 图片名称
                txt_file_name = image['file_name'].split('.')[0] + '.txt' # 要保存的对应txt文件名
                break
        # print(width,height,img_file_name,txt_file_name)
        segmentation = annotation['segmentation'] # 图像分割点信息[[x1,y1,x2,y2,...,xn,yn]]
        seg_x_y_list = [i/width if num%2==0 else i/height for num,i in enumerate(segmentation[0])] # 归一化图像分割点信息
        label_seg_x_y_list = seg_x_y_list[:]
        label_seg_x_y_list.insert(0,category_id-1) # 图像类别与分割点信息[label,x1,y1,x2,y2,...,xn,yn]
        # print(label_seg_x_y_list)

        # 写txt文件
        txt_file_path = target_dir + txt_file_name
        # print(txt_file_path)
        write_yolo_txt_file(txt_file_path,label_seg_x_y_list)

        # 选出txt对应img文件
        img_file_path = img_dir + img_file_name
        # print(img_file_path)
        shutil.copy(img_file_path,target_dir)



if __name__=="__main__":
    img_dir = 'Javeri/valid/'
    target_dir = 'Javeri/valid1/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    in_json_path = 'Javeri/valid/_annotations.coco.json'
    read_json(in_json_path,img_dir,target_dir)
