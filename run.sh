# get dataset statistics
python src/get_dataset_stat.py

# run training
mkdir -p ./results

HIDDEN_DIM=128 bash shell/train_rnn.sh
HIDDEN_DIM=256 bash shell/train_rnn.sh
HIDDEN_DIM=512 bash shell/train_rnn.sh

HIDDEN_DIM=128 bash shell/train_cnn.sh
HIDDEN_DIM=256 bash shell/train_cnn.sh
HIDDEN_DIM=512 bash shell/train_cnn.sh

# run analysis
bash shell/analysis.sh
