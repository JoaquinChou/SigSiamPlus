# class balance DRSN for classification with unsup constrastive learning
# python main_sigsiam.py --batch_size 1024 --learning_rate 0.1  --temp 0.8 --cosine --dataset spectra_quest_signal --epoch 1500
python main_sigsiam.py --batch_size 1024 --learning_rate 0.1  --temp 0.7 --cosine --dataset spectra_quest_signal --epoch 1000 --model DRSN-SA > temp_1500.txt
python main_sigsiam.py --batch_size 1024 --learning_rate 0.1  --temp 0.7 --cosine --dataset CWRU_signal_cross_domain --epoch 1000 --model DRSN-SA > temp.txt

# for plot tsne
python gen_label_feature_source_target.py  --results_txt 03-18-13-49_last --model DRSN-SA
python TSNE_source_target.py --initial_dims 256 --results_txt 03-18-13-49_last

python gen_label_feature_sigsiam.py  --results_txt 03-17-15-04_last --model DRSN-SA
python TSNE.py --initial_dims 256 --results_txt 03-17-15-04_last
python gen_label_feature_sigsiam.py  --results_txt 03-12-22-48_last --model DRSN-SA --dataset CWRU_signal_cross_domain

# for the constrasting learning to train the linear layer_____fine_tuning
# 更改模型路径和数据集路径即可
python bottleneck_main_linear.py --fine_tuning  > temp.txt
python bottleneck_main_linear.py --model DRSN-SA  --dataset CWRU_signal_cross_domain --fine_tuning --class_num 4 > temp.txt
python bottleneck_main_linear.py --model DRSN-SA  --dataset spectra_quest_signal --fine_tuning --class_num 4 --is_pretrained > temp.txt
python bottleneck_main_linear.py --model DRSN-SA --dataset CWRU_signal_cross_domain --fine_tuning  --class_num 4  --is_pretrained > temp.txt
python bottleneck_main_linear.py --model DRSN-SA --dataset SEU_bearing --fine_tuning  --class_num 4 --is_pretrained > 20_classifier.txt

python bottleneck_test_linear.py --model DRSN-SA  --dataset spectra_quest_signal --class_num 4
python bottleneck_test_linear.py --model DRSN-SA  --dataset CWRU_signal --class_num 7
python bottleneck_test_linear.py --model DRSN-SA  --dataset CWRU_signal_cross_domain --class_num 4



python shot_main_sigisiam.py --dataset spectra_quest_signal --class_num 4 > temp.txt
python shot_main_sigisiam.py --dataset CWRU_signal_cross_domain --class_num 4 > temp.txt

