import train
if __name__ == "__main__":
    cat_dim = (10,)
    noise_dim = (60,)
    batch_size = 32
    n_batch_per_epoch = 2000
    nb_epoch = 1000
    #start to train
    train.train(cat_dim,noise_dim,batch_size,n_batch_per_epoch,nb_epoch)