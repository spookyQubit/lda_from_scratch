import numpy as np


class LDA(object):
    def __init__(self,
                 documents,
                 vocab,
                 n_topics,
                 logger,
                 alpha=1,
                 beta=1,
                 random_seed=42):
        """
        :param documents: list of docs. Each doc is a list of vocabs
                          d1 = [v1, v1, v1, v2, v1, v1, v2]
                          d2 = [v2, v2, v1, v2, v1, v1, v2]
                          documents = [d1, d2]
        :param vocab: list of unique words which form the vocab for the documents.
                      This could have been derived from the documents itself but
                      for simplicity, we will assume that it will be provided by the user.
                      v1 = "a"
                      v2 = "b"
                      vocab = [v1, v2]
        :param n_topics: int
        :param alpha: float. Hyper-parameter for the Diritchlet prior for document's topic probability
        :param beta: float. Hyper-parameter for the Diritchlet prior each topics's word probability
        :param iterations: int. Number of times the iterations should be run for Gibbs sampling
        :param burn_in: int. Number of first iters to disregard
        """

        self.D = documents
        self.n_docs = len(documents)
        self.V = vocab
        self.n_vocab = len(vocab)
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.logger = logger
        self.random_seed = random_seed

        # The counts which need to be kept to perform collapsed Gibbs sampling
        # The notation is to match the notes
        # N_0v.,  N_1v.   --> N_vt = {v1: {t0: count_t0, t1: count_t1},
        #                             v2: {t0: count_t0, t1: count_t1}}
        # N_0..,  N_1..   --> N_t  = {t0: count_t0, t1: count_t1 }
        # N_0.di, N_1.di  --> N_dt = {d1: {t0: count_t0, t1: count_t1},
        #                             d2: {t0: count_t0, t1: count_t1}}
        # N_..di          --> N_d  = {d1: count_d1, d2: count_d2}

        self.N_vt = {v: {t_id: 0 for t_id in range(self.n_topics)} for v in self.V}
        self.N_t = {t_id: 0 for t_id in range(self.n_topics)}
        self.N_d = {d_id: len(d) for d_id, d in enumerate(self.D)}
        self.N_dt = {d_id: {t_id: 0 for t_id in range(self.n_topics)} for d_id in range(len(self.D))}

        self.L = []
        self.randomly_init_counts()

        self.p_docs_t_dist = {}
        self.p_topics_v_dist = {}

    def randomly_init_counts(self):

        np.random.seed(self.random_seed)

        for d_id, d in enumerate(self.D):
            topic_assignments = []
            for w_id, w in enumerate(d):
                t_id = np.random.randint(self.n_topics)
                topic_assignments.append(t_id)

                self.N_vt[w][t_id] += 1
                self.N_t[t_id] += 1
                self.N_dt[d_id][t_id] += 1

            self.L.append(topic_assignments)

        self.logger.info("N_vt = {}".format(self.N_vt))
        self.logger.info("N_t = {}".format(self.N_t))
        self.logger.info("N_d = {}".format(self.N_d))
        self.logger.info("N_dt = {}".format(self.N_dt))

    def get_topic_for_each_word(self, iterations):
        for _ in range(iterations):
            for d_id, d in enumerate(self.D):
                for w_id, w in enumerate(d):
                    current_t_id = self.L[d_id][w_id]
                    self.N_dt[d_id][current_t_id] -= 1
                    self.N_d[d_id] -= 1
                    self.N_t[current_t_id] -= 1
                    self.N_vt[w][current_t_id] -= 1
                    p_t = []
                    for t_id in range(self.n_topics):

                        left_numerator = self.beta + self.N_vt[w][t_id]
                        left_denominator = self.n_vocab * self.beta + self.N_t[t_id]
                        left = left_numerator/left_denominator

                        right_numerator = self.alpha + self.N_dt[d_id][t_id]
                        right_denominator = self.n_topics * self.alpha + self.N_d[d_id]
                        right = right_numerator/right_denominator

                        p_t.append(left * right)
                    p_t = [pt / sum(p_t) for pt in p_t]

                    # New assignment
                    new_t_id = np.random.multinomial(1, p_t).argmax()

                    self.L[d_id][w_id] = new_t_id

                    self.N_dt[d_id][new_t_id] += 1
                    self.N_d[d_id] += 1
                    self.N_t[new_t_id] += 1
                    self.N_vt[w][new_t_id] += 1

        self.logger.info("L = {}".format(self.L))
        return self.L

    def get_topic_dist_of_docs(self):
        for d_id, d in enumerate(self.D):
            p_t_dist = []
            for t_id in range(self.n_topics):
                numerator = self.N_dt[d_id][t_id] + self.alpha
                denominator = self.N_d[d_id] + self.n_topics * self.alpha
                p_t_dist.append(numerator/denominator)
            self.p_docs_t_dist[d_id] = p_t_dist

        self.logger.info("p_docs_t_dist = {}".format(self.p_docs_t_dist))
        return self.p_docs_t_dist

    def get_vocab_dist_of_topics(self):
        for t_id in range(self.n_topics):
            p_v_dist = []
            for v in self.V:
                numerator = self.N_vt[v][t_id] + self.beta
                denominator = self.N_t[t_id] + self.n_vocab * self.beta
                p_v_dist.append(numerator/denominator)
            self.p_topics_v_dist[t_id] = p_v_dist

        self.logger.info("p_topics_v_dist = {}".format(self.p_topics_v_dist))
        return self.p_topics_v_dist
