python3 main.py --experiment isic_bnn_1 --task test_bnn --seed 1111;
python3 main.py --experiment isic_cnn_1 --task test_cnn+ --seed 1111;
python3 main.py --experiment isic_label_smoothing_0.1_1 --task test_cnn+ --label_smoothing 0.1 --seed 1111;
python3 main.py --experiment isic_label_smoothing_0.2_1 --task test_cnn+ --label_smoothing 0.2 --seed 1111;
python3 main.py --experiment isic_focal_loss_2.0_1 --task test_cnn+ --focal_loss True --focal_gamma 2.0 --seed 1111;
python3 main.py --experiment isic_focal_loss_5.0_1 --task test_cnn+ --focal_loss True --focal_gamma 5.0 --seed 1111;

python3 main.py --experiment isic_bnn_2 --task test_bnn --seed 2222;
python3 main.py --experiment isic_cnn_2 --task test_cnn+ --seed 2222;
python3 main.py --experiment isic_label_smoothing_0.1_2 --task test_cnn+ --label_smoothing 0.1 --seed 2222;
python3 main.py --experiment isic_label_smoothing_0.2_2 --task test_cnn+ --label_smoothing 0.2 --seed 2222;
python3 main.py --experiment isic_focal_loss_2.0_2 --task test_cnn+ --focal_loss True --focal_gamma 2.0 --seed 2222;
python3 main.py --experiment isic_focal_loss_5.0_2 --task test_cnn+ --focal_loss True --focal_gamma 5.0 --seed 2222;

python3 main.py --experiment isic_bnn_3 --task test_bnn --seed 3333;
python3 main.py --experiment isic_cnn_3 --task test_cnn+ --seed 3333;
python3 main.py --experiment isic_label_smoothing_0.1_3 --task test_cnn+ --label_smoothing 0.1 --seed 3333;
python3 main.py --experiment isic_label_smoothing_0.2_3 --task test_cnn+ --label_smoothing 0.2 --seed 3333;
python3 main.py --experiment isic_focal_loss_2.0_3 --task test_cnn+ --focal_loss True --focal_gamma 2.0 --seed 3333;
python3 main.py --experiment isic_focal_loss_5.0_3 --task test_cnn+ --focal_loss True --focal_gamma 5.0 --seed 3333;

python3 main.py --experiment pcam_bnn_1 --task test_bnn --seed 1111 --dataset pcam --dataset_dir ../../Datasets/pcam --image_x 96 --image_y 96 --batch_size 32;
python3 main.py --experiment pcam_cnn_1 --task test_cnn+ --seed 1111 --dataset pcam --dataset_dir ../../Datasets/pcam --image_x 96 --image_y 96 --batch_size 32;
python3 main.py --experiment pcam_label_smoothing_0.1_1 --task test_cnn+ --label_smoothing 0.1 --seed 1111 --dataset pcam --dataset_dir ../../Datasets/pcam --image_x 96 --image_y 96 --batch_size 512;
python3 main.py --experiment pcam_label_smoothing_0.2_1 --task test_cnn+ --label_smoothing 0.2 --seed 1111 --dataset pcam --dataset_dir ../../Datasets/pcam --image_x 96 --image_y 96 --batch_size 512;
python3 main.py --experiment pcam_focal_loss_2.0_1 --task test_cnn+ --focal_loss True --focal_gamma 2.0 --seed 1111 --dataset pcam --dataset_dir ../../Datasets/pcam --image_x 96 --image_y 96 --batch_size 512;
python3 main.py --experiment pcam_focal_loss_5.0_1 --task test_cnn+ --focal_loss True --focal_gamma 5.0 --seed 1111 --dataset pcam --dataset_dir ../../Datasets/pcam --image_x 96 --image_y 96 --batch_size 512;
