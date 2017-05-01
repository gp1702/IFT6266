import lasagne
import theano
import theano.tensor as T
import text_utils
import pickle as pkl
import utils
import numpy as np
import os

#theano.config.compute_test_value = 'warn'

class Model(object):
    def __init__(self, options, it=None):
        self.options = options
        self.it = it

    def initialise(self):

        network_fn = self.build_network()
        self.network = network_fn()

    def reload(self, model_file):

        # Reloading
        options = pkl.load(open(model_file + '.pkl'))

        self.options = options
        network_fn = self.build_network()
        network = network_fn()
        print "reloading {}...".format(model_file)
        self.network = utils.load_model(network, model_file)

    def build_network(self):
        raise NotImplementedError


class GAN(Model):

    def __init__(self, options, it = None):
        super(GAN, self).__init__(options, it)
        self.schedule = 0.
        self.last_loss = [0,0]

    def initialise(self):
        generator_fn, discriminator_fn = self.build_network()

        self.generator = generator_fn()
        self.discriminator = discriminator_fn()

    def reload(self, model_dir):
        
        print "reloading..."
        model_file = os.path.join(model_dir, 'model')

        self.options = pkl.load(open(model_file + '.pkl'))
        self.initialise()
        
        # Reloading the generator
        with np.load(model_file + "_generator.npz") as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            print [x.shape for x in param_values]
            lasagne.layers.set_all_param_values(self.generator, param_values)

        # Reloading th ediscriminator
        with np.load(model_file + "_discriminator.npz") as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.discriminator, param_values)


    def save(self, model_dir):

        model_file = os.path.join(model_dir, 'model')

        print "len:", [x.shape for x in lasagne.layers.get_all_param_values(self.generator)]

        np.savez(model_file+"_generator.npz", *lasagne.layers.get_all_param_values(self.generator))
        np.savez(model_file+"_discriminator.npz", *lasagne.layers.get_all_param_values(self.discriminator))
        option_file = model_file + '.pkl'
        pkl.dump(self.options, open(option_file, 'w'))

    def build_generator(self, input_var=None):
        filter_dimension = self.options['generator_filter_dimension']
        noise_dimension = self.options['noise_dimension']

        print "We have {} hidden units".format(filter_dimension)

        network = lasagne.layers.InputLayer(shape=(None, noise_dimension),
                                            input_var=input_var)

        network = lasagne.layers.ReshapeLayer(network, (-1, noise_dimension, 1, 1))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension, filter_size=(4, 4),
                                                       stride=(1, 1)))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension/2,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2, output_size=8))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension/4,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2, output_size=16))

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid, 
                                                       output_size=32)
        
        return network

    def build_discriminator(self, input_var=None):
        filter_dimension = self.options['discriminator_filter_dimension']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "We have {} hidden units".format(filter_dimension)

        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)

        network = lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                             nonlinearity=lasagne.nonlinearities.sigmoid)

        return network

    def build_network(self):
        return self.build_generator, self.build_discriminator

    def calculate_loss(self, discriminator_score_real_image, discriminator_score_generated_image, real_img, generated_image):

        
        use_wgan = self.options['use_wgan']
        l2_penalty = self.options['l2_penalty']
        gan_penalty = self.options['gan_penalty']

        if use_wgan:
            print "Using the wgan loss"
            discriminator_loss=-0.5*((discriminator_score_real_image-discriminator_score_generated_image).mean())
            gan_generator_loss=-0.5*(discriminator_score_generated_image.mean())
        else:
            discriminator_loss = -(T.log(discriminator_score_real_image) + T.log(1.-discriminator_score_generated_image)).mean()
            gan_generator_loss = -T.log(discriminator_score_generated_image).mean()

        l2_loss = (l2_penalty)*lasagne.objectives.squared_error(generated_image, real_img).mean()
        generator_loss = gan_penalty*gan_generator_loss + l2_loss
        return discriminator_loss, generator_loss, [gan_penalty*gan_generator_loss, l2_loss]
    
    def _generator_output(self, input_vars):
        return lasagne.layers.get_output(self.generator, *input_vars)
    
    def _discriminator_output(self, generated_image, inputs_var, real=True): 
        return lasagne.layers.get_output(self.discriminator, generated_image)

    def compile_theano_func(self, learning_rate):
        # Prepare Theano variables for inputs and targets
        inputs, input_vars = self._get_inputs()
        origin_real_img = T.tensor4('real_img')

        real_img = origin_real_img.dimshuffle((0, 3, 1, 2))

        # Generator output
        generated_image = self._generator_output(input_vars) 

        #Discriminator
        discriminator_score_real_image = self._discriminator_output(real_img, input_vars, real=True) #score for the real image
        discriminator_score_generated_image = self._discriminator_output(generated_image, input_vars, real=False) # score for the generated image

        # loss
        discriminator_loss, generator_loss, debug_values = self.calculate_loss(discriminator_score_real_image, discriminator_score_generated_image, real_img, generated_image)

        # generator update
        weights_generator = lasagne.layers.get_all_params(self.generator, trainable=True)
        weight_updates_generator = lasagne.updates.adam(generator_loss, weights_generator, learning_rate=learning_rate, beta1=0.5)

        #Discriminator update
        weights_discriminator = lasagne.layers.get_all_params(self.discriminator, trainable=True)
        weight_updates_discriminator = lasagne.updates.adam(discriminator_loss, weights_discriminator, learning_rate=learning_rate, beta1=0.5)



        self.train_generator_fn = None
        self.train_discriminator_fn = None
        self.train_generator_fn = theano.function(inputs + [origin_real_img], [generator_loss]+debug_values,
                                                  updates=weight_updates_generator,
                                                  allow_input_downcast=True) 

        self.train_discriminator_fn = theano.function(inputs + [origin_real_img], [discriminator_loss],
                                                  updates=weight_updates_discriminator,
                                                  allow_input_downcast=True) # inputs can be either noise+bagel or only noise or only caps , original_real_img = always the tim_bit(generated by the generator) 

        
        self.generate_sample_fn = theano.function(inputs, [generated_image.transpose((0, 2, 3, 1))],
                                      allow_input_downcast=True)

    def _get_inputs(self):

        #TODO add noise

        input = T.matrix('noise')
        input_var = input#input.transpose((0, 3, 1, 2))
        return [input], [input_var]

    def train(self, imgs, target, caps):

        noise_dimension = self.options['noise_dimension']
        noise = np.random.normal(size=(len(imgs), noise_dimension))

        use_wgan = self.options['use_wgan']
        
        [disc_loss] = self.train_discriminator_fn(noise, target)
        gen_loss = self.train_generator_fn(noise, target)

        if use_wgan:
            discriminator_params_values=lasagne.layers.get_all_param_values(self.discriminator, trainable=True)
            clamped_weights= [np.clip(w, -0.05, 0.05) for  w in discriminator_params_values]
            lasagne.layers.set_all_param_values(self.discriminator, clamped_weights, trainable=True)
        
        return disc_loss, gen_loss

    def get_generation_fn(self):
        
        noise_dimension = self.options['noise_dimension']
        
        def val_fn(imgs, target, caps):
            
            noise = np.random.uniform(size=(len(imgs), noise_dimension))
            res = self.generate_sample_fn(noise)
            return 0, res[0]

        return val_fn

class Variational_GAN(GAN):
    def build_generator(self, contour=None, noise=None):
        filter_dimension = self.options['generator_filter_dimension']
        noise_dimension = self.options['noise_dimension']

        
        print "Building generator"
        #Our encoder
        self._contour_input = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                            input_var=contour)
        
        #encoder = lasagne.layers.BatchNormLayer(self._contour_input)
        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(self._contour_input, num_filters=filter_dimension/8, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension/4, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension/2, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.FlattenLayer(encoder)
        

        self._noise_input = lasagne.layers.InputLayer(shape=(None, noise_dimension),
                                            input_var=noise)
        
        # Merging the encoder and the noise.
        network = lasagne.layers.ConcatLayer([self._noise_input, encoder])

        network = lasagne.layers.ReshapeLayer(network, (-1, noise_dimension + filter_dimension*4*4, 1, 1))

        network = lasagne.layers.batch_norm(
            lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension, filter_size=(4, 4),
                                                 stride=(1, 1)))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 2,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=8))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 4,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=16))

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid,
                                                       output_size=32)


        return network

    def build_discriminator(self, input_var=None):
        filter_dimension = self.options['discriminator_filter_dimension']
        encoder_size = self.options['discriminator_encoder_size']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "building discriminator"
        #print "We have {} hidden units".format(filter_dimension)

        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                            input_var=input_var)
        
        #We have one aditionnal layer.
        
        network = lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/16, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/8, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                             nonlinearity=lasagne.nonlinearities.sigmoid)

        return network
    
    def _generator_output(self, input_vars):
        return lasagne.layers.get_output(self.generator,
                                             {self._noise_input:input_vars[0], self._contour_input:input_vars[1]})
    
    def _discriminator_output(self, generated_image, inputs_var, real=True):
        
        # We set the middle
        contour = inputs_var[1] 
        center = (contour.shape[2] / 2, contour.shape[3] / 2)
        
        contour = T.set_subtensor(contour[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16],  generated_image)
        
        return lasagne.layers.get_output(self.discriminator, contour)
    
    def _get_inputs(self):

        noise = T.matrix('noise')
        contour = T.tensor4('contour') # defining a variable called contour this is similar to initializing int x in C.
        
        contour_var = contour.transpose((0, 3, 1, 2)) # fed directly to generator
        return [noise, contour], [noise, contour_var]
    
    def train(self, imgs, target, caps): # imgs = bagel, target = tim-bit, caps = captions 
    
        noise_dimension = self.options['noise_dimension']
        noise = np.random.uniform(size=(len(imgs), noise_dimension))
        use_wgan = self.options['use_wgan']
        
        [disc_loss] = self.train_discriminator_fn(noise, imgs, target) # inherited from class -> GAN , noise = noise , imgs = contour, target = real_images.
        self.last_loss[0] = disc_loss

        if False or self.schedule == 0:
            rval = self.train_generator_fn(noise, imgs, target)
            self.last_loss[1] = rval

        self.schedule = (self.schedule + 1) % 2
        
        if use_wgan:
            discriminator_params_values=lasagne.layers.get_all_param_values(self.discriminator, trainable=True)
            clamped_weights= [np.clip(w, -0.05, 0.05) for  w in discriminator_params_values]
            lasagne.layers.set_all_param_values(self.discriminator, clamped_weights, trainable=True)

        return self.last_loss
        #return [disc_loss, gen_loss]

    def get_generation_fn(self):
        
        noise_dimension = self.options['noise_dimension']
        
        def val_fn(imgs, target, caps):
            
            noise = np.random.uniform(size=(len(imgs), noise_dimension))
            res = self.generate_sample_fn(noise, imgs)

            return 0, res[0]

        return val_fn

class Hybrid_GAN(Variational_GAN):
    def build_generator(self, contour=None, noise=None):
        filter_dimension = self.options['generator_filter_dimension']
        encoder_size = self.options['generator_encoder_size']
        noise_dimension = self.options['noise_dimension']

        print "We have {} hidden units".format(filter_dimension)

        # Our encoder
        self._contour_input = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                                        input_var=contour)

        # encoder = lasagne.layers.BatchNormLayer(self._contour_input)
        encoder = lasagne.layers.batch_norm(
            lasagne.layers.Conv2DLayer(self._contour_input, num_filters=filter_dimension / 8, filter_size=(5, 5),
                                       stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(
            lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension / 4, filter_size=(5, 5),
                                       stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(
            lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension / 2, filter_size=(5, 5),
                                       stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(
            lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension, filter_size=(5, 5),
                                       stride=2, pad=2))

        encoder = lasagne.layers.FlattenLayer(encoder)

        self._noise_input = lasagne.layers.InputLayer(shape=(None, noise_dimension),
                                                      input_var=noise)

        # Merging the encoder and the noise.
        network = lasagne.layers.ConcatLayer([self._noise_input, encoder])

        network = lasagne.layers.ReshapeLayer(network, (-1, noise_dimension + filter_dimension * 4 * 4, 1, 1))

        network = lasagne.layers.batch_norm(
            lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension, filter_size=(4, 4),
                                                 stride=(1, 1)))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 2,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=8))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 4,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=16))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 8,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=32))

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid,
                                                       output_size=64)

        return network

    def _generator_output(self, input_vars):
        return lasagne.layers.get_output(self.generator,
                                         {self._noise_input: input_vars[0], self._contour_input: input_vars[1]})

    def _discriminator_output(self, generated_image, inputs_var, real=True):

        if real:
            # We set the middle
            contour = inputs_var[1]
            contour = utils.reconstruct(contour, generated_image,flag = 1)

            return lasagne.layers.get_output(self.discriminator, contour)

        else:
            # For this model, it is already 64x64
            return lasagne.layers.get_output(self.discriminator, generated_image)

    def calculate_loss(self, discriminator_score_real_image, discriminator_score_generated_image, real_img, generated_image):

        # WGAN loss
        use_wgan = self.options['use_wgan']
        l2_penalty = self.options['l2_penalty']
        gan_penalty = self.options['gan_penalty']

        if use_wgan:
            print "Using the wgan loss"
            discriminator_loss = -0.5 * ((discriminator_score_real_image - discriminator_score_generated_image).mean())
            gan_generator_loss = -0.5 * (discriminator_score_generated_image.mean())
        else:
            discriminator_loss = -(T.log(discriminator_score_real_image) + T.log(1. - discriminator_score_generated_image)).mean()
            gan_generator_loss = -T.log(discriminator_score_generated_image).mean()

        # The L2 loss
        center = (generated_image.shape[2] / 2, generated_image.shape[3] / 2)
        center_sample = generated_image[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]
        center_img = real_img
        l2_loss = (l2_penalty) * lasagne.objectives.squared_error(center_sample, center_img).mean()
        generator_loss = gan_penalty*gan_generator_loss + l2_loss
        return discriminator_loss, generator_loss, [gan_penalty*gan_generator_loss, l2_loss]

class GAN_Captions(Variational_GAN):

    def __init__(self, options, it = None):
        super(GAN_Captions, self).__init__(options, it)

        #self.caps_model = caps_model(self.options, self.it)
        self._caps = None
        self._contour_input = None
        self.W = None

    def build_generator(self, contour = None, caps= None, input_var=None):
        filter_dimension = self.options['filter_dimension']
        emb_size = self.options['emb_size']
        vocab_size = self.options['vocab_size']
        rnn_size = self.options['rnn_size']
        use_bag_of_word = self.options['use_bag_of_word']
        noise_dimension = self.options['noise_dimension']

        if use_bag_of_word:
            #print "Using a neural bag of words."
            self._caps = lasagne.layers.InputLayer(shape=(None, emb_size), input_var=caps)
            network = lasagne.layers.DenseLayer(self._caps, rnn_size)

        else:
            #print "Using a recurrent nnet."
            W = self.get_emb()
            self._caps = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=caps)
            network = lasagne.layers.EmbeddingLayer(self._caps, vocab_size, emb_size, W=W)
            network = lasagne.layers.GRULayer(network, rnn_size, only_return_final=True)

        network2 = lasagne.layers.DenseLayer(network,5)

        print "Filter bank depth= {} ".format(filter_dimension)
        self._contour_input = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),input_var=contour)
        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(self._contour_input, num_filters=filter_dimension / 8, filter_size=(5, 5),stride=2, pad=2))
        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension / 4, filter_size=(5, 5),stride=2, pad=2))
        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension / 2, filter_size=(5, 5),stride=2, pad=2))
        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=filter_dimension, filter_size=(5, 5),stride=2, pad=2))
        encoder = lasagne.layers.FlattenLayer(encoder)



        #self._noise_input = lasagne.layers.InputLayer(shape=(None, noise_dimension),
        #                                              input_var=noise)

        # Merging the encoder and the noise.
        network = lasagne.layers.ConcatLayer([network2, encoder])

        network = lasagne.layers.ReshapeLayer(network, (-1, noise_dimension + filter_dimension * 4 * 4, 1, 1))

        network = lasagne.layers.batch_norm(
            lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension, filter_size=(4, 4),
                                                 stride=(1, 1)))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 2,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=8))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 4,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=16))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension / 8,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=32))

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid,
                                                       output_size=64)

        return network


    def _generator_output(self, input_vars):
        return lasagne.layers.get_output(self.generator, {self._caps:input_vars[0], self._contour_input:input_vars[1]})


    def train(self, imgs, target, caps):  # imgs = bagel, target = tim-bit, caps = captions     

        noise_dimension = self.options['noise_dimension']
        noise = np.random.normal(size=(len(imgs), noise_dimension))
        caps = [cap[np.random.choice(len(cap))] for cap in caps]
        caps = utils.pad_to_the_max(caps)


        [disc_loss] = self.train_discriminator_fn(noise,imgs, target) # inherited from class->GAN i.e captions , 
        gen_loss = self.train_generator_fn(caps, imgs, target) # inherited from class->GAN, train the generator on captions plus the contour 

        use_wgan = self.options['use_wgan']
        if use_wgan:
            discriminator_params_values = lasagne.layers.get_all_param_values(self.discriminator, trainable=True)
            clamped_weights = [np.clip(w, -0.05, 0.05) for w in discriminator_params_values]
            lasagne.layers.set_all_param_values(self.discriminator, clamped_weights, trainable=True)

        return disc_loss, gen_loss

    def get_generation_fn(self):
        noise_dimension = self.options['noise_dimension']

        def val_fn(imgs, target, caps):
            caps = [cap[np.random.choice(len(cap))] for cap in caps]
            caps = utils.pad_to_the_max(caps)

            res = self.generate_sample_fn(caps,imgs)
            return 0, res[0]

        return val_fn

    def build_discriminator(self, input_var=None):
        filter_dimension = self.options['discriminator_filter_dimension']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "Filter bank depth = {} ".format(filter_dimension)

        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                            input_var=input_var)

        network = lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=filter_dimension, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                             nonlinearity=lasagne.nonlinearities.sigmoid)

        return network

    def _get_inputs(self):

        contour = T.tensor4('contour') # **chnage**defining a variable called contour this is similar to initializing int x in C.
        
        contour_var = contour.transpose((0, 3, 1, 2)) # **chnage** fed directly to generator
        #return [noise, contour], [noise, contour_var]
        input = T.imatrix('captions') #initialize a tensor with the dimensions of the caption size i.e. vocab_size*emb_size
        rval = input

        use_bag_of_word = self.options['use_bag_of_word']
        if use_bag_of_word:
            #Get the embeddings and sum.
            W = self.get_emb()
            rval = W[input]
            rval = rval.sum(axis=1) # shape = (vocab_size,emb_size)

        return [input, contour], [rval, contour_var]
    

    def get_emb(self):


        if self.W is None:
            vocab_size = self.options['vocab_size']
            emb_file = self.options['emb_file']
            emb_size = self.options['emb_size']

            self.W = np.random.normal(loc=0.0, scale=0.01, size=(vocab_size, emb_size)).astype('float32')


            # Loading the embeding file.
            if emb_file is not None:

                print "We have pretrained embedings."
                it = self.it

                nb_present = 0
                for i, line in enumerate(open(emb_file)):
                    line = line.split(' ')
                    word = line[0]
                    emb = [float(x) for x in line[1:]]

                    if word in it.vocab:
                        self.W[it.mapping[word]] = emb
                        nb_present += 1

                    if i % 100000 == 0:
                        print "Done {} words".format(i)

                print "There were {} on {} words present.".format(nb_present, len(it.vocab))

            self.W = theano.shared(self.W)

        return self.W

    def _discriminator_output(self, generated_image, inputs_var, real=True):
        if real:
            # We set the middle
            contour = inputs_var[1]
            contour = utils.reconstruct(contour, generated_image, flag = 1)

            return lasagne.layers.get_output(self.discriminator, contour)
        else:
            return lasagne.layers.get_output(self.discriminator, generated_image)

    def calculate_loss(self, discriminator_score_real_image, discriminator_score_generated_image, real_img, generated_image):

        # WGAN loss
        use_wgan = self.options['use_wgan']
        l2_penalty = self.options['l2_penalty']
        gan_penalty = self.options['gan_penalty']

        if use_wgan:
            print "Using the wgan loss"
            discriminator_loss = -0.5 * ((discriminator_score_real_image - discriminator_score_generated_image).mean())
            gan_generator_loss = -0.5 * (discriminator_score_generated_image.mean())
        else:
            discriminator_loss = -(T.log(discriminator_score_real_image) + T.log(1. - discriminator_score_generated_image)).mean()
            gan_generator_loss = -T.log(discriminator_score_generated_image).mean()

        # The L2 loss
        center = (generated_image.shape[2] / 2, generated_image.shape[3] / 2)
        center_sample = generated_image[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]
        center_img = real_img
        l2_loss = (l2_penalty) * lasagne.objectives.squared_error(center_sample, center_img).mean()
        generator_loss = gan_penalty*gan_generator_loss + l2_loss
        return discriminator_loss, generator_loss, [gan_penalty*gan_generator_loss, l2_loss]

class Text_GAN(GAN):

    def __init__(self, options, it = None):
        super(Text_Image_GAN, self).__init__(options, it)
        

    def build_generator(self, input_var=None):

        filter_dimension = self.options['filter_dimension']
        emb_size = self.options['emb_size']
        vocab_size = self.options['vocab_size']
        rnn_size = self.options['rnn_size']
        use_bag_of_word = self.options['use_bag_of_word']

        network = lasagne.layers.InputLayer(shape=(None, emb_size), input_var=input_var)
        network = lasagne.layers.DenseLayer(network, rnn_size)
        #W = self.get_emb()
        #network = lasagne.layers.EmbeddingLayer(network, vocab_size, emb_size, W=W)
        network = lasagne.layers.GRULayer(network,rnn_size,only_return_final=True)


        # fully connected
        network = lasagne.layers.ReshapeLayer(network, (-1, rnn_size, 1, 1))

        # Deconv
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension*2, filter_size=(7, 7), stride=(1, 1))
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=filter_dimension, filter_size=(4, 4), stride=(2,2))
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(2, 2), stride=(2,2), nonlinearity=lambda x: x.clip(0., 1.))

        
        return network

    def get_emb(self):
        self.W = None

        if self.W is None:
            vocab_size = self.options['vocab_size']
            emb_file = self.options['emb_file']
            emb_size = self.options['emb_size']

            self.W = np.random.normal(loc=0.0, scale=0.01, size=(vocab_size, emb_size)).astype('float32')


            # Loading the embeding file.
            if emb_file is not None:

                print "We have pretrained embedings."
                it = self.it

                nb_present = 0
                for i, line in enumerate(open(emb_file)):
                    line = line.split(' ')
                    word = line[0]
                    emb = [float(x) for x in line[1:]]

                    if word in it.vocab:
                        self.W[it.mapping[word]] = emb
                        nb_present += 1

                    if i % 100000 == 0:
                        print "Done {} words".format(i)

                print "There were {} on {} words present.".format(nb_present, len(it.vocab))

            self.W = theano.shared(self.W)

        return self.W


    def _get_inputs(self):
        input = T.imatrix('captions') #initialize a tensor with the dimensions of the caption size i.e. 7854*100
        rval = input
        W = self.get_emb()
        rval = W[input]
        rval = rval.sum(axis=1)
        return [input], [rval]

    def train(self, imgs, target, caps):  # imgs = bagel, target = tim-bit, caps = captions     


        caps = [cap[np.random.choice(len(cap))] for cap in caps]
        caps = text_utils.pad_to_the_max(caps)


        [disc_loss] = self.train_discriminator_fn(caps, target) # inherited from class->GAN i.e captions , 
        gen_loss = self.train_generator_fn(caps, target) # inherited from class->GAN

        use_wgan = self.options['use_wgan']
        if use_wgan:
            discriminator_params_values = lasagne.layers.get_all_param_values(self.discriminator, trainable=True)
            clamped_weights = [np.clip(w, -0.05, 0.05) for w in discriminator_params_values]
            lasagne.layers.set_all_param_values(self.discriminator, clamped_weights, trainable=True)

        return disc_loss, gen_loss

    def get_generation_fn(self):
        noise_dimension = self.options['noise_dimension']

        def val_fn(imgs, target, caps):
            caps = [cap[np.random.choice(len(cap))] for cap in caps]
            caps = text_utils.pad_to_the_max(caps)

            res = self.generate_sample_fn(caps)
            return 0, res[0]

        return val_fn
