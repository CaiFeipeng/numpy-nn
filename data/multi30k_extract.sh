
mkdir multi30k
wget https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz  -O ./multi30k/training.tar.gz --no-check-certificate
wget https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz -O ./multi30k/validation.tar.gz --no-check-certificate
wget https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz  -O ./multi30k/mmt16_task1_test.tar.gz --no-check-certificate


tar -xvf ./multi30k/training.tar.gz -C ./multi30k
tar -xvf ./multi30k/validation.tar.gz -C ./multi30k
tar -xvf ./multi30k/mmt16_task1_test.tar.gz -C ./multi30k