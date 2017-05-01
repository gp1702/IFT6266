#modified form https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import utils
import lasagne
import process_data
import pickle as pkl
import models




def load_dataset(batch_size=128, load_caption=False):

    batch_train = process_data.process(batch_size=batch_size, extract_center=True, load_caption=load_caption)
    batch_val = process_data.inital_process(nb_sub=2000, batch_size=batch_size, img_path = 'val2014', extract_center=True, load_caption=load_caption)


    try:
        batch_val.vocab = batch_train.vocab
        batch_val.mapping = batch_train.mapping
        batch_val.process_captions()
    except Exception as e:
        print "Captions not processed"
        print e

    return batch_train, batch_val



def train(GAN_type, num_epochs=20,
          learning_rate=0.001, sample=3, save_freq=100,
          batch_size=128, verbose_freq=100,
          model_dir="models/testin123/",
          reload=False,
          load_caption = False, return_64_64=True, show_imgs = False,
          **kwargs):

    # Load the dataset
    print "Loading data..."
    batch_train, batch_val = load_dataset(batch_size, load_caption=load_caption)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

   
    val_loss = []
    train_loss = []

    my_model = GAN_type(kwargs, batch_train)
    # Reloading
    if reload:
        my_model.reload(model_dir)
    else:
        my_model.initialise()

    my_model.compile_theano_func(learning_rate=learning_rate)

    print "Starting training..."
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print "Running epoch", epoch +1
        
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i, batch in enumerate(batch_train):
            inputs, targets, caps = batch
            losses = my_model.train(inputs, targets, caps)
            #train_err += train_err_tmp
            train_batches += 1


            # Generate
            if (i+1) % verbose_freq == 0.:
                #Get the images
                figs = utils.generate_and_show_sample(my_model.get_generation_fn(), nb=sample, seed=i, it=batch_val, n_split=2, return_64_64=return_64_64)

                for fig_no, fig in enumerate(figs):
                    fig_name = os.path.join(model_dir, "epoch_{}_batch_{}_split_{}.jpg".format(epoch, i, fig_no))

                    #save it
                    fig.savefig(fig_name)
                    if show_imgs:
                        fig.show()

                print "batch {} of epoch {} of {} took {:.3f}s".format(i, epoch + 1, num_epochs, time.time() - start_time)
                #print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
                print "losses:", losses

            if (i+1) % save_freq == 0:
                print "saving the model to", model_dir
                my_model.save(model_dir)
                print "losses:", losses

        train_loss.append(train_err)
        print "Training Complete.."

    return my_model



if __name__ == '__main__':
 
    import models

    
    filter_dimension = 512

    my_model = train(models.GAN_Captions, learning_rate=0.0002, num_epochs=250, sample=10,
                           save_freq=600, verbose_freq=100000, batch_size=128, reload=False,
                           model_dir="models/Conditonal_GAN/",
                           generator_filter_dimension=filter_dimension,
                           load_caption=True,
                           generator_encoder_size=filter_dimension,
                           discriminator_filter_dimension=filter_dimension,
                           discriminator_encoder_size=filter_dimension,
                           filter_dimension=filter_dimension,
                           emb_size=100, vocab_size=7574, rnn_size=filter_dimension, use_bag_of_word=True,
                           emb_file=None,
                           noise_dimension=5, use_wgan=True,
                           l2_penalty=32, gan_penalty=0.01,
                           return_64_64=True,
                           show_imgs=True
                           )

    it_train, it_valid = load_dataset(load_caption=True)    
    utils.generate_and_show_sample(my_model.get_generation_fn(), nb=20, seed=1993, it=it_valid, verbose=True, n_split=4, return_64_64=False)
