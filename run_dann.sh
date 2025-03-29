python ./dann.py \
--log /u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/ham_vs_msk_loc_head_neck.log \
--phase test \
--seed 42 \
--train-resizing res. \
--scale 1.0 \
--ratio 1.0 \
--no-hflip \
--resize-size 224 \
--norm-mean 0.5 0.5 0.5 \
--norm-std 0.5 0.5 0.5 \
--data ISIC \
--source ham_loc_head_neck \
--target msk_loc_head_neck \
--epochs 50 \
--lr 0.01 \
--i 250 \
/u/student/2021/cs21resch15002/DomainShift/Datasets/ISIC_Dataset

