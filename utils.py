

from process_data import inital_process
import matplotlib.pyplot as plt
import numpy as np
import lasagne
import theano
from theano import tensor as T
import pickle as pkl
from collections import OrderedDict, Counter
import numpy as np

def reconstruct(img, middle,flag = 1):
    
    if flag ==1:
        center = (img.shape[2] / 2, img.shape[3] / 2)
        img = T.set_subtensor(img[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16],
                              middle)
    else:
        center = (int(np.floor(img.shape[1] / 2.)), int(np.floor(img.shape[2] / 2.)))
        img = np.copy(img)
        img[:, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = middle
    return img

def generate_and_show_sample(fn, nb=1, seed=1993, it=None, verbose=True, n_split=1, return_64_64=False):

    if it is None:
        it = inital_process(img_path="val2014", load_caption=False, process_text=True)

    choice = range(len(it))
    if seed > 0:
        np.random.seed(seed)
        np.random.shuffle(choice)

    choice = choice[:nb] * 5

    #try:
    xs, ys, cs = zip(*[it[i] for i in choice])
    loss, preds = fn(xs, ys, cs)

    for pl in np.array_split(np.arange(nb), n_split):
            show_sample([xs[i] for i in pl], [ys[i] for i in pl], [preds[i] for i in pl], len(pl), return_64_64=return_64_64)
    

    try:
        if verbose and it.mapping is not None:
            for img in cs:
                sentence = [it.mapping[idx] for idx in img[0]]
                print ' '.join(sentence)
                print ""
    except AttributeError:
        pass

def get_theano_generative_func(network_path, network_fn):



    input = T.tensor4('inputs')
    target = T.tensor4('targets')

    input_var = input.transpose((0, 3, 1, 2))
    target_var = target.dimshuffle((0, 3, 1, 2))

    network = network_fn(input_var)
    network = load_model(network, network_path)

    test_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    print "Computing the functions..."
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_prediction.transpose((0, 2, 3, 1))])
    return val_fn

def show_sample(xs, ys, preds, nb=1, return_64_64=False):

    for i in range(nb):
        img_true = np.copy(xs[i])
        center = (int(np.floor(img_true.shape[0] / 2.)), int(np.floor(img_true.shape[1] / 2.)))

        img_true[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = ys[i]

        plt.subplot(2, nb, i+1)
        plt.axis('off')
        plt.imshow(img_true)
        
        if not return_64_64:
            img_pred = np.copy(xs[i])
            img_pred[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = preds[i]
            plt.subplot(2, nb, nb+i+1)
            #plt.axis('off')
            plt.imshow(img_pred)
        else:
            #plt.axis('off')
            plt.subplot(2, nb, nb+i+1)
            plt.imshow(preds[i])
            
            
    plt.show()



def save_model(network, options, file_name):
    np.savez(file_name, *lasagne.layers.get_all_param_values(network))
    option_file = file_name + '.pkl'
    pkl.dump(options, open(option_file, 'w'))



def load_model(network, file_name):
    with np.load(file_name) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network

def remove_punctuations(sentence):
    # Do a bit of steaming
    sentence = sentence.replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('\'', '').replace \
        ('"', '')
    sentence = sentence.lower().split()
    return sentence

def get_vocab(data, nb_words=50000, min_nb=10, remove_stop_words = True):
    """
    Get the vocabulary and the mapping (int to string) of the captions
    :param data:
    :param nb_words:
    :param min_nb:
    :param remove_stop_words:
    :return:
    """


    # Put everything into onw long string
    data = [item for sublist in list(data.values()) for item in sublist]
    data = " ".join(data)

    # Do a bit of steaming
    data = remove_punctuations(data)
    vocab = Counter(data)

    # Remove the stop words
    new_vocab = vocab.copy()
    for key, value in vocab.items():
        if remove_stop_words and key in stopwords:
            del new_vocab[key]
        if value < min_nb:
            del new_vocab[key]

    vocab = new_vocab

    # Keep the most common words
    vocab = Counter(dict(vocab.most_common(nb_words)))

    # Extract a mapping
    mapping = {}
    mapping[1] = "--UNK--"
    mapping["--UNK--"] = 1
    for i, word in enumerate(sorted(vocab.keys())):
        mapping[i + 2] = word
        mapping[word] = i + 2

    return vocab, mapping


def filter_caps(data, mapping, switch=False):

    """
    Filter the data (remove unknown words, switch to ints, etc...)
    :param data:
    :param mapping:
    :param switch:
    :return:
    """

    data_filtered = {}
    for img_name in data:
        tmp = []
        for cap in data[img_name]:
            words = remove_punctuations(cap)
            if switch:
                filtered = [mapping[word] if word in mapping else mapping["--UNK--"] for word in words]
            else:
                filtered = [word if word in mapping else "--UNK--" for word in words]

            tmp.append(filtered)
        data_filtered[img_name] = tmp

    return data_filtered

def pad_to_the_max(data):

    max_len = max([len(x) for x in data])
    data = [x + [0] * (max_len - len(x)) for x in data]
    return np.array(data)


stopwords ="""a
about
above
after
again
against
all
am
an
and
any
are
aren't
as
at
be
because
been
before
being
below
between
both
but
by
can't
cannot
could
couldn't
did
didn't
do
does
doesn't
doing
don't
down
during
each
few
for
from
further
had
hadn't
has
hasn't
have
haven't
having
he
he'd
he'll
he's
her
here
here's
hers
herself
him
himself
his
how
how's
i
i'd
i'll
i'm
i've
if
in
into
is
isn't
it
it's
its
itself
let's
me
more
most
mustn't
my
myself
no
nor
not
of
off
on
once
only
or
other
ought
our
ours
ourselves
out
over
own
same
shan't
she
she'd
she'll
she's
should
shouldn't
so
some
such
than
that
that's
the
their
theirs
them
themselves
then
there
there's
these
they
they'd
they'll
they're
they've
this
those
through
to
too
under
until
up
very
was
wasn't
we
we'd
we'll
we're
we've
were
weren't
what
what's
when
when's
where
where's
which
while
who
who's
whom
why
why's
with
won't
would
wouldn't
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves""".split("\n")