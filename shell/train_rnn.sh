for EMBEDDING_DIM in 300
do
    for DATASET_NAME in imdb yelp twitter
    do
        for MODEL_NAME in rnn lstm
        do
            python main.py --dataset $DATASET_NAME --model $MODEL_NAME \
                --embed_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM
        done
    done
done

python ./src/get_performance.py
python ./src/get_performance_table.py
