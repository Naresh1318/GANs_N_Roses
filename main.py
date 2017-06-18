from ops import *
from utils import *
import os
import time
import datetime
import mission_control as mc
from scipy.misc import imsave as ims
from tensorflow.contrib.layers import batch_norm


def discriminator(image, reuse=False):
    """
    Used to distinguish between real and fake images.
    :param image: Images feed to the discriminate.
    :param reuse: Set this to True to allow the weights to be reused.
    :return: A logits value.
    """
    df_dim = 64
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv'))
    h1 = lrelu(batch_norm(conv2d(h0, 64, df_dim * 2, name='d_h1_conv'),
                          center=True, scale=True, is_training=True, scope='d_bn1'))
    h2 = lrelu(batch_norm(conv2d(h1, 128, df_dim * 4, name='d_h2_conv'),
                          center=True, scale=True, is_training=True, scope='d_bn2'))
    h3 = lrelu(batch_norm(conv2d(h2, 256, df_dim * 8, name='d_h3_conv'),
                          center=True, scale=True, is_training=True, scope='d_bn3'))
    h4 = dense(tf.reshape(h3, [mc.BATCH_SIZE, -1]), 4 * 4 * 512, 1, scope='d_h3_lin')
    return h4


def generator(z, z_dim):
    """
    Used to generate fake images to fool the discriminator.
    :param z: The input random noise.
    :param z_dim: The dimension of the input noise.
    :return: Fake images -> [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]
    """
    gf_dim = 64  # TODO: Implement different image size.
    z2 = dense(z, z_dim, gf_dim * 8 * 4 * 4, scope='g_h0_lin')
    h0 = tf.nn.relu(batch_norm(tf.reshape(z2, [-1, 4, 4, gf_dim * 8]),
                               center=True, scale=True, is_training=True, scope='g_bn1'))
    h1 = tf.nn.relu(batch_norm(conv_transpose(h0, [mc.BATCH_SIZE, 8, 8, gf_dim * 4], "g_h1"),
                               center=True, scale=True, is_training=True, scope='g_bn2'))
    h2 = tf.nn.relu(batch_norm(conv_transpose(h1, [mc.BATCH_SIZE, 16, 16, gf_dim * 2], "g_h2"),
                               center=True, scale=True, is_training=True, scope='g_bn3'))
    h3 = tf.nn.relu(batch_norm(conv_transpose(h2, [mc.BATCH_SIZE, 32, 32, gf_dim * 1], "g_h3"),
                               center=True, scale=True, is_training=True, scope='g_bn4'))
    h4 = conv_transpose(h3, [mc.BATCH_SIZE, 64, 64, 3], "g_h4")
    return tf.nn.tanh(h4)


def form_results():
    """
    Forms a folder for each run and returns the path of the folders formed
    :return: path of the folders created
    """
    path = './Results/{}/'.format(mc.DATASET_CHOSEN)
    results_folder = '{0}_{1}_{2}_{3}_{4}_{5}' \
        .format(datetime.datetime.now(), mc.Z_DIM, mc.BATCH_SIZE, mc.N_ITERATIONS, mc.LEARNING_RATE, mc.BETA_1)
    results_path = path + results_folder
    tensorboard_path = results_path + '/Tensorboard'
    generated_images_path = results_path + '/Generated_Images'
    saved_models_path = results_path + '/Saved_Models'

    if not os.path.exists(path + results_folder):
        os.mkdir(results_path)
        os.mkdir(generated_images_path)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_models_path)
    return results_path, tensorboard_path, generated_images_path, saved_models_path


def train(z_dim, batch_size, learning_rate, beta1, n_iter, image_size):
    """
    Function used to train a DCGAN
    :param z_dim: Dimension of the input noise which will be feed as the input to the generator.
    :param batch_size: Batch size of the images to train on.
    :param learning_rate: Learning rate for both the Generator and the Discriminator.
    :param beta1: The exponential decay rate for the 1st moment estimates.
    :param n_iter: The number of iterations to train the GAN on.
    :param image_size: Dimension of the images to be created.
    """
    # Create a folder for this run under the Results folder
    results_path, tensorboard_path, generated_images_path, saved_models_path = form_results()

    # Size of the image to  be formed
    imageshape = [image_size, image_size, 3]

    start_time = time.time()

    # Read the images from the database
    real_img = load_dataset(mc.DATASET_PATH, data_set=mc.DATASET_CHOSEN, image_size=image_size)

    # Placeholders to pass the image and the noise to the network
    images = tf.placeholder(tf.float32, [batch_size] + imageshape, name="real_images")
    zin = tf.placeholder(tf.float32, [None, z_dim], name="z")

    G = generator(zin, z_dim)  # G(z)
    Dx = discriminator(images)  # D(x)
    Dg = discriminator(G, reuse=True)  # D(G(x))

    # Loss
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, targets=tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, targets=tf.zeros_like(Dg)))
    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, targets=tf.ones_like(Dg)))
    dloss = d_loss_real + d_loss_fake

    # Get the variables which need to be trained
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(dloss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(gloss, var_list=g_vars)

    # Used to save the model
    saver = tf.train.Saver(max_to_keep=5)

    # Used as the input to D to display fake images
    display_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

    logdir = tensorboard_path
    tf.summary.scalar('Discriminator Loss', dloss)
    tf.summary.scalar('Generator Loss', gloss)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
        if mc.TRAIN:
            for idx in range(n_iter):
                batch_images = next_batch(real_img, batch_size=10)
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

                for k in range(1):
                    sess.run([d_optim], feed_dict={images: batch_images, zin: batch_z})
                for k in range(1):
                    sess.run([g_optim], feed_dict={zin: batch_z})

                print("[%4d/%4d] time: %4.4f, " % (idx, n_iter, time.time() - start_time))

                if idx % 10 == 0:
                    # Display the loss and run tf summaries
                    summary = sess.run(summary_op, feed_dict={images: batch_images, zin: batch_z})
                    writer.add_summary(summary, global_step=idx)
                    d_loss = d_loss_fake.eval({zin: display_z, images: batch_images})
                    g_loss = gloss.eval({zin: batch_z})
                    print("\n Discriminator loss: {0} \n Generator loss: {1}".format(d_loss, g_loss))

                if idx % 200 == 0:
                    # Save the generated images every 200 iterations
                    sdata = sess.run([G], feed_dict={zin: display_z})
                    print(np.shape(sdata))
                    ims(generated_images_path + '/' + str(idx) + ".jpg", merge(sdata[0], [3, 4]))

                if idx % 1000 == 0:
                    saver.save(sess, saved_models_path + "/train", global_step=idx)
        else:
            # TODO: Get the latest saved model path and change the saved_models_path accordingly
            saver.restore(sess, tf.train.latest_checkpoint(saved_models_path + "/"))
            batch_z = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
            batch_z = np.repeat(batch_z, batch_size, axis=0)
            for i in range(z_dim):
                edited = np.copy(batch_z)
                edited[:, i] = (np.arange(0.0, batch_size) / (batch_size / 2)) - 1
                sdata = sess.run([G], feed_dict={zin: edited})
                ims("results/imagenet/" + str(i) + ".jpg", merge(sdata[0], [3, 4]))


train(z_dim=mc.Z_DIM, batch_size=mc.BATCH_SIZE, n_iter=mc.N_ITERATIONS,
      learning_rate=mc.LEARNING_RATE, beta1=mc.BETA_1, image_size=mc.IMAGE_SIZE)
