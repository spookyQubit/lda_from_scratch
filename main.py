import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging
import sys

from lda import LDA


def get_logger():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename="/home/shantanu/PycharmProjects/lda_topic_modeling/lda.log",
                        level=logging.INFO)
    logger = logging.getLogger("LDA")
    return logger


def save_image(p_topics_v_dist, i, n_topics, img_shape):
    for t_id in range(n_topics):
        t_data = 256 * np.reshape(np.array(p_topics_v_dist[t_id]), img_shape)
        fig = plt.imshow(t_data, 'gray', origin='lower', interpolation='none', vmin=0, vmax=256)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig('images/' + 'image_iter_' + str(i) + '_id_' + str(t_id) + '.jpg')


def plot_convergence(n_topics):
    fig = plt.figure(figsize=(40, 50))
    number_of_images = 8
    for i in range(int(number_of_images / n_topics)):
        for t_id in range(n_topics):
            image_file_path = 'images/' + 'image_iter_' + str(5 * i) + '_id_' + str(t_id) + '.jpg'
            img = cv2.imread(image_file_path)
            ax = fig.add_subplot(int(number_of_images/n_topics), n_topics, n_topics * i + 1 + t_id)
            ax.set_title("it = {}, id = {}".format(5 * i, t_id), fontsize=50)
            ax.grid(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='both',
                           bottom=False, labelbottom=False,
                           left=False, labelleft=False)
            ax.imshow(img)

    fig.savefig('./images/topics_v_dist.jpg')
    plt.show()


def main():

    # Vocab
    v1 = "a"
    v2 = "b"
    v3 = "y"
    v4 = "z"
    V = [v1, v2, v3, v4]

    # Topics
    n_topics = 2

    # Documents
    d1 = [v1, v1, v1, v1, v1, v1, v2]
    d2 = [v2, v1, v1, v1, v1, v1, v3]
    d3 = [v3, v3, v4, v3, v3, v3, v4, v3, v2, v2, v2, v2]
    d4 = [v4, v4, v4, v3, v3, v4, v4, v2, v2, v2, v2]
    D = [d1, d2, d3, d4]

    # Hyper parameters
    alpha = 1  # Pseudo counts for topic selection
    beta = 1  # Pseudo counts for vocab selection given a particular topic

    lda = LDA(documents=D,
              vocab=V,
              n_topics=n_topics,
              logger=get_logger(),
              alpha=alpha,
              beta=beta,
              random_seed=92)

    # Creating images and saving them

    p_topics_v_dist = lda.get_vocab_dist_of_topics()
    save_image(p_topics_v_dist, 0, n_topics=n_topics, img_shape=(2, int(len(V)/2)))

    total_iterations = 50
    for i in range(1, total_iterations):
        lda.get_topic_for_each_word(iterations=1)

        lda.get_topic_dist_of_docs()
        p_topics_v_dist = lda.get_vocab_dist_of_topics()

        if i % 5 == 0:
            print("i = {}".format(i))
            save_image(p_topics_v_dist, i, n_topics=n_topics, img_shape=(2, int(len(V)/2)))

    plot_convergence(n_topics)


if __name__ == "__main__":
    sys.exit(main())

