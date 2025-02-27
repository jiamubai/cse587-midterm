for EMBEDDING_DIM in 300
do
    for DATASET_NAME in imdb yelp twitter
    do
        if EMBEDDING_DIM == 50
        then
            KERNEL_SIZE_LIST=(3 5 7)
        else
            KERNEL_SIZE_LIST=(7)
        fi

        for KERNEL_SIZE in ${KERNEL_SIZE_LIST[@]}
        do
            python main.py --dataset $DATASET_NAME --model cnn \
                --embed_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM \
                --cnn_kernel_size $KERNEL_SIZE
        done
    done
done

python ./src/get_performance.py
python ./src/get_performance_table.py
