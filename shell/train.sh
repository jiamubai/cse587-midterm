for DATASET_NAME in imdb yelp twitter
do
    for MODEL_NAME in rnn lstm
    do
        python main.py --dataset $DATASET_NAME --model $MODEL_NAME \
            --hidden_dim $HIDDEN_DIM
    done

    for KERNEL_SIZE in 3 5 7
    do
        python main.py --dataset $DATASET_NAME --model cnn \
            --hidden_dim $HIDDEN_DIM --kernel_size $KERNEL_SIZE
    done
done
