7. 训练
首先是数据准备，分为三步。

1）annotation file

在 ./data/my_data/ 目录下生成 train.txt/val.txt/test.txt 文件。txt 文件中一行表示一张图片，形式为：图片绝对路径 + box_1 + box_2 + … + box_n。Box 的形式为：label_index + x_min + y_min + x_max + y_max，原始坐标为图片左上角。

例如：

xxx/xxx/1.jpg 0 453 369 473 391 1 588 245 608 268
xxx/xxx/2.jpg 1 466 403 485 422 2 793 300 809 320
…

注意：每个 txt 文件最后一行为空白行。

2）class_names file

在 ./data/my_data/ 目录下生成 data.names 文件，每一行代表一个类别名称。例如：

bird
person
bike
…

3）prior anchor file

使用 kMeans 算法来选择 anchor boxes：

python get_kmeans.py
然后，你将得到 9 个 anchors 和评价 IOU，把 anchors 保存在 txt 文件中。

准备完数据之后就可以开始训练了。

使用 train.py 文件，函数参数如下：

$ python train.py -h
usage: train.py [-h] [--train_file TRAIN_FILE] [--val_file VAL_FILE]
               [--restore_path RESTORE_PATH] 
               [--save_dir SAVE_DIR]
               [--log_dir LOG_DIR] 
               [--progress_log_path PROGRESS_LOG_PATH]
               [--anchor_path ANCHOR_PATH]
               [--class_name_path CLASS_NAME_PATH] [--batch_size BATCH_SIZE]
               [--img_size [IMG_SIZE [IMG_SIZE ...]]]
               [--total_epoches TOTAL_EPOCHES]
               [--train_evaluation_freq TRAIN_EVALUATION_FREQ]
               [--val_evaluation_freq VAL_EVALUATION_FREQ]
               [--save_freq SAVE_FREQ] [--num_threads NUM_THREADS]
               [--prefetech_buffer PREFETECH_BUFFER]
               [--optimizer_name OPTIMIZER_NAME]
               [--save_optimizer SAVE_OPTIMIZER]
               [--learning_rate_init LEARNING_RATE_INIT] [--lr_type LR_TYPE]
               [--lr_decay_freq LR_DECAY_FREQ]
               [--lr_decay_factor LR_DECAY_FACTOR]
               [--lr_lower_bound LR_LOWER_BOUND]
               [--restore_part [RESTORE_PART [RESTORE_PART ...]]]
               [--update_part [UPDATE_PART [UPDATE_PART ...]]]
               [--update_part [UPDATE_PART [UPDATE_PART ...]]]
               [--use_warm_up USE_WARM_UP] [--warm_up_lr WARM_UP_LR]
               [--warm_up_epoch WARM_UP_EPOCH]
8. 评价
使用 eval.py 来评估验证集和测试集，函数参数如下：

$ python eval.py -h
usage: eval.py [-h] [--eval_file EVAL_FILE] [--restore_path RESTORE_PATH]
              [--anchor_path ANCHOR_PATH] 
              [--class_name_path CLASS_NAME_PATH]
              [--batch_size BATCH_SIZE]
              [--img_size [IMG_SIZE [IMG_SIZE ...]]]
              [--num_threads NUM_THREADS]
              [--prefetech_buffer PREFETECH_BUFFER]
函数返回 loss、召回率 recall、精准率 precision，如下所示：

recall: 0.927, precision: 0.945
total_loss: 0.210, loss_xy: 0.010, loss_wh: 0.025, loss_conf: 0.125, loss_class: 0.050

9. 其它技巧
训练的时候可以尝试使用下面这些技巧：

Data augmentation：使用 ./utils/data_utils.py 中的 data_augmentation 方法来增加数据。
像 Gluon CV 一样混合和 label 平滑。

正则化技巧，例如 L2 正则化。

多尺度训练：你可以像原稿中的作者那样定期改变输入图像的尺度（即不同的输入分辨率）。
