import os
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg


class darknet_read(object):
    def __init__(self, phase, rebuild=False):
        self.data_path = cfg.DATA_PATH
        self.train_path = os.path.join(self.data_path, 'tr_data')
        self.test_path = os.path.join(self.data_path, 'test_data')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(
            self.cache_path, 'darknet_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            annot_path = os.path.join(self.data_path, 'tr_data')
        else:
            annot_path = os.path.join(self.data_path, 'test_data')
        #get files
        lf=os.listdir(annot_path)
        lf.sort()
        jpeg_files = [x for x in lf if '.JPEG' in x]
        txt_files = [x for x in lf if '.txt' in x]
        
        gt_labels = []
        for i in range(0,len(jpeg_files)):
        #for jpeg_file,txt_file in annotnnot_list:
            label, num = self.load_darknet_annotation(jpeg_files[i],txt_files[i])
            if num == 0:
                continue
            imname = os.path.join(annot_path, jpeg_files[i])
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

   


    def load_darknet_annotation(self, index):
        """
        derived from load_pascal_annotation()
        load image and txt file
        recalc the bound box coordinates
        darknet format: class_index| center_of_boundbox_x/width| center_of_boundbox_y/height| size_of_bb_x/width|size_of_bb_y|height
        recaculated into xmin,ymin,xmax,ymax

        """
        image_size = cfg.IMAGE_SIZE

        cell_size = cfg.CELL_SIZE
        #print('isize:',image_size)
        #imname = '/content/yolo_tensorflow/data/gasmeter/tr_data/idd0001.JPEG'
        imname = os.path.join(self.data_path, 'idd', index + '.JPEG')

        im = cv2.imread(imname)
        h_ratio = 1.0 * image_size / im.shape[0]
        w_ratio = 1.0 * image_size / im.shape[1]
        # im = cv2.resize(im, [image_size, image_size])

        label = np.zeros((cell_size,cell_size, 25))
        #filename ='/content/yolo_tensorflow/data/gasmeter/tr_data/idd0001.txt'
        annot_name = os.path.join(self.data_path, 'idd', index + '.txt')
        ff=open(annot_name)

        objs=0
        for obj in ff: #typically single line, anyway, multiple lines with multiple object can theoretically happen
            objs+=1
            bb = obj.split(' ')
            # Make pixel indexes 0-based
            #bb[0] is class id
            x1 = float(bb[1])*image_size #darknet annotations are in percentage of picture width or height so transforming to TF annot should just *input layer size
            y1 = float(bb[2])*image_size
            x2 = float(bb[3])*image_size
            y2 = float(bb[4])*image_size
            print(x1)
            #print(h_ratio,w_ratio)
            #print(x1,x2,y1,y2)
            cls_ind = int(bb[0]) #self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            #print(boxes)
            x_ind = int(boxes[0] * cell_size / image_size)
            y_ind = int(boxes[1] * cell_size / image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, objs
