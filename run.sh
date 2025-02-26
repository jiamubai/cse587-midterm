for DATASET_NAME in imdb # yelp twitter
do
    for MODEL_NAME in cnn #  rnn lstm
    do
        python main.py --dataset $DATASET_NAME --model $MODEL_NAME
    done
done
