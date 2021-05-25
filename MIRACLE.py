import numpy as np
np.set_printoptions(suppress=True)
import random
import tensorflow.compat.v1 as tf
from sklearn.metrics import mean_squared_error
tf.disable_v2_behavior() 
tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(linewidth=np.inf)

class MIRACLE(object):
    def __init__(self, num_train, lr  = 0.001, batch_size = 32, num_inputs = 1, num_outputs = 1,
                 w_threshold = 0.3, n_hidden = 32, hidden_layers = 2, ckpt_file = 'tmp.ckpt',
                 standardize = True,  reg_lambda=1, reg_beta=1, DAG_min = 0.5, debug = False,
                 DAG_only = False, missing_list = range(1),reg_m = 1.0, window = 10, max_steps = 400):
        n_indicators = len(missing_list)
        print("Number of Indicators = ", n_indicators)
        '''
        Assume n_indicators is the number of variables at the RHS of adjacency matrix.
        '''
        self.missing_list = missing_list
        self.w_threshold = w_threshold
        self.DAG_min = DAG_min
        self.learning_rate = lr
        self.reg_lambda = reg_lambda
        self.reg_beta = reg_beta
        self.reg_m = reg_m
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.X = tf.placeholder("float", [None, self.num_inputs])
        self.X_mask = tf.placeholder("float", [None, self.num_inputs])
        self.rho =  tf.placeholder("float",[1,1])
        self.alpha =  tf.placeholder("float",[1,1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.window = window       
        self.max_steps = max_steps
        self.metric = mean_squared_error
        self.debug = debug
        self.missing_value = -1
        self.count = 0

        
        # One-hot vector indicating which nodes are trained
        self.sample =tf.placeholder(tf.int32, [self.num_inputs])


        
        # Store layers weight & bias
        seed = 1
        self.weights = {}
        self.biases = {}

        # Create the input and output weight matrix for each feature
        for i in range(self.num_inputs):
            self.weights['w_h0_' + str(i)] = tf.Variable(
                tf.random_normal([self.num_inputs, self.n_hidden], seed=seed) * 0.01)
            self.weights['out_' + str(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.num_outputs], seed=seed))

            # self.weights['w_h0_' + str(i)][:,-n_indicators] *= 0.0
        for i in range(self.num_inputs):
            self.biases['b_h0_' + str(i)] = tf.Variable(tf.random_normal([self.n_hidden], seed=seed) * 0.01)
            self.biases['out_' + str(i)] = tf.Variable(tf.random_normal([self.num_outputs], seed=seed))

        # The first and second layers are shared
        self.weights.update({
            'w_h1': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden])),
            'w_h2': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
        })

        self.biases.update({
            'b_h1': tf.Variable(tf.random_normal([self.n_hidden])),
            'b_h2': tf.Variable(tf.random_normal([self.n_hidden]))
        })


        self.hidden_h0 = {}
        self.hidden_h1 = {}
        self.hidden_h2 = {}

        self.layer_1 = {}
        self.layer_1_dropout = {}
        self.out_layer = {}
       
        self.Out_0 = []
        
        # Mask removes the feature i from the network that is tasked to construct feature i
        self.mask = {}
        self.activation = tf.nn.elu


        indices = [self.num_inputs - n_indicators] * self.n_hidden
        indicator_mask = tf.transpose(tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))
        for i in range(self.num_inputs - n_indicators + 1, self.num_inputs):
            indices = [i] * self.n_hidden
            indicator_mask *= tf.transpose(tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))
            
        for i in range(self.num_inputs):
            indices = [i]*self.n_hidden
            self.mask[str(i)] = tf.transpose(tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))* indicator_mask
            
            self.weights['w_h0_'+str(i)] = self.weights['w_h0_'+str(i)]*self.mask[str(i)]
            self.hidden_h0['nn_'+str(i)] = self.activation(tf.add(tf.matmul(self.X, self.weights['w_h0_'+str(i)]), self.biases['b_h0_'+str(i)]))
            self.hidden_h1['nn_'+str(i)] = self.activation(tf.add(tf.matmul(self.hidden_h0['nn_'+str(i)], self.weights['w_h1']), self.biases['b_h1']))
            self.out_layer['nn_'+str(i)] = tf.matmul(self.hidden_h1['nn_'+str(i)], self.weights['out_'+str(i)]) + self.biases['out_'+str(i)]
            self.Out_0.append(self.out_layer['nn_'+str(i)])
        
        # Concatenate all the constructed features
        self.Out = tf.concat(self.Out_0,axis=1)
        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
           
        # Residuals
        self.R = self.X - self.Out 
        self.R_proxy = []
        self.R_proxy = tf.square(tf.multiply(self.X[:,:-n_indicators],self.X_mask[:,:-n_indicators]) - tf.multiply(self.Out[:,:-n_indicators],self.X_mask[:,:-n_indicators]))
        self.supervised_loss =  tf.reduce_sum(self.R_proxy)
        self.reg_moment = 0
        self.X_replace = []

        for dim in range(n_indicators):
            index = missing_list[dim]
            prob = (tf.nn.sigmoid(self.Out[:, self.num_inputs-n_indicators+dim])+1.01)/1.02
            weight = 1.0/(prob)

            a = tf.reduce_mean(self.Out[:, index]) #* self.X_mask[:, index])/num_obs
            b = tf.reduce_sum(self.X[:, index]*self.X_mask[:, index]*weight)/tf.reduce_sum(weight)
            self.reg_moment += tf.square(a-b)

        bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.X[:,-n_indicators:], logits = self.Out[:,-n_indicators:])
        self.R_proxy = tf.concat([self.R_proxy, bce_loss], axis = 1)


        self.regularization_loss = 0

        self.W_0 = []
        for i in range(self.num_inputs):
            self.W_0.append(tf.math.sqrt(tf.reduce_sum(tf.square(self.weights['w_h0_'+str(i)]),axis=1,keepdims=True)))
        
        self.W = tf.concat(self.W_0,axis=1)
               
        #truncated power series
        d = tf.cast(self.X.shape[1], tf.float32)
        coff = 1.0 
        Z = tf.multiply(self.W,self.W)
       
        dag_l = tf.cast(d, tf.float32) 
       
        Z_in = tf.eye(d)
        for i in range(1,25):
           
            Z_in = tf.matmul(Z_in, Z)
           
            dag_l += 1./coff * tf.linalg.trace(Z_in)
            coff = coff * (i+1)
        
        self.h = dag_l - tf.cast(d, tf.float32)



        #group lasso
        L1_loss = 0.0
        for i in range(self.num_inputs):
            if i< self.num_inputs-n_indicators:
                w_1 = tf.slice(self.weights['w_h0_'+str(i)],[0, 0],[i, -1])
                w_2 = tf.slice(self.weights['w_h0_'+str(i)],[i+1, 0],[self.num_inputs-n_indicators-(i+1),-1])
                L1_loss += tf.reduce_sum(tf.norm(w_1,axis=1)) + tf.reduce_sum(tf.norm(w_2,axis=1))
            else:
                w_1 = tf.slice(self.weights['w_h0_'+str(i)],[0, 0],[self.num_inputs-n_indicators, -1])
                L1_loss += tf.reduce_sum(tf.norm(w_1,axis=1))
        
        # Divide the residual into untrain and train subset
        _, subset_R = tf.dynamic_partition(tf.transpose(self.R_proxy), partitions=self.sample, num_partitions=2)
        subset_R = tf.transpose(subset_R)

        self.mse_loss_subset =  tf.reduce_mean(tf.reduce_sum(subset_R,axis=1))

        self.regularization_loss_subset =  self.mse_loss_subset +  self.reg_beta * L1_loss +  0.5 * self.rho * self.h * self.h + self.alpha * self.h + self.reg_m * self.reg_moment
            
        if DAG_only:
            self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset)
        else:
            self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset + self.Lambda *self.rho* self.supervised_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())     
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.tmp = ckpt_file
        
    def __del__(self):
        tf.reset_default_graph()
        print("Destructor Called... Cleaning up")
        self.sess.close()
        del self.sess
        

    def fit(self, X_missing, X_mask, X_truth,num_nodes, X_seed = None, early_stopping = False):         
        X = X_missing.copy()
        print("Using hyperparameters (lambda, beta, reg_m, lr): ", self.reg_lambda, self.reg_beta, self.reg_m, self.learning_rate)
        from random import sample 
        rho_i = np.array([[1]])
        alpha_i = np.array([[1.0]])
        best_loss = 1e9
        
        one_hot_sample = [0]*self.num_inputs
        subset_ = sample(range(self.num_inputs),num_nodes) 
        for j in subset_:
            one_hot_sample[j] = 1
        
        if len(X_seed) == 1:
            self.missing_value = [int((np.nanmax(X) - np.nanmin(X)) * 2)] * self.num_inputs
            inds = np.where(np.isnan(X))
            X = np.array(X)
            X[inds] = np.take(self.missing_value, inds[1]) 
        elif X_seed is not None:
            X = X_seed
        else:
            col_mean = np.nanmean(X, axis = 0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1]) 
    
        avg_seed = []
        
    
        for step in range(1, self.max_steps):
            

            if step == 0 or (step % 1 == 0 and step > self.window):

                X_pred = np.mean(np.array(avg_seed), axis = 0)
                X =  X * X_mask + X_pred * (1 - X_mask)
                if step % 10 == 0:
                    print("step", step, "of", self.max_steps, "steps" )

            h_value, loss = self.sess.run([self.h, self.supervised_loss], feed_dict={self.X: X, 
                                          self.X_mask:X_mask, self.keep_prob : 1, self.rho:rho_i, 
                                          self.alpha:alpha_i, self.is_train : True, self.noise:0})
            if self.debug:        
                print("Step " + str(step) + ", Loss= " + "{:.4f}".format(loss)," h_value:", h_value) 

            idxs = np.arange(X.shape[0])
            random.shuffle(idxs)
            for step1 in range(1, (X.shape[0] // self.batch_size) + 1):     
                batch_x = X[idxs[step1*self.batch_size:step1*self.batch_size + self.batch_size]]
                batch_x_masks = X_mask[step1*self.batch_size:step1*self.batch_size + self.batch_size]
                self.sess.run(self.loss_op_dag, feed_dict={self.X: batch_x, self.X_mask:batch_x_masks, self.sample:one_hot_sample,
                                                              self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.Lambda : 
                                                                  self.reg_lambda, self.is_train : True, self.noise : 0})

            if early_stopping:
                if loss < best_loss and step > 10:
                    self.saver.save(self.sess, self.tmp)
                    best_loss = loss
                
            if self.debug:
                '''
                Note these values are and SHOULD BE used for diagnostic purposes ONLY!!!
                '''
                print("In-Dist RMSE = ", self.rmse_loss(X_truth, self.pred(X), X_mask))

            
            if len(avg_seed) >= self.window: 
                avg_seed.pop(0)
            avg_seed.append(self.pred(X))
            
            
            
        if early_stopping:
            self.saver.restore(self.sess, self.tmp)

        
        if self.debug:
            print(X_truth)
            print(self.pred(X))


        X_pred = np.mean(np.array(avg_seed), axis = 0)
        return X * X_mask + X_pred * (1 - X_mask), X_pred, avg_seed

    
    def rmse_loss(self, ori_data, imputed_data, data_m):
        numerator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
        denominator = np.sum(1-data_m)
        return np.sqrt(numerator/float(denominator))
    

    def pred(self, X):
        return self.sess.run(self.Out, feed_dict={self.X: X, self.keep_prob: 1, self.is_train: False, self.noise: 0})

    def get_weights(self, X):
        return self.sess.run(self.W, feed_dict={self.X: X,  self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
    
    def pred_W(self, X):
        W_est = self.sess.run(self.W, feed_dict={self.X: X, self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
        return np.round_(W_est,decimals=3)
