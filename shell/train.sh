for EMBEDDING_DIM in 300
do
    for DATASET_NAME in imdb yelp twitter
    do
        for MODEL_NAME in rnn lstm
        do
            python main.py --dataset $DATASET_NAME --model $MODEL_NAME \
                --embed_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM
        done

        for KERNEL_SIZE in 3 5 7
        do
            python main.py --dataset $DATASET_NAME --model cnn \
                --embed_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM \
                --cnn_kernel_size $KERNEL_SIZE
        done
    done
done
