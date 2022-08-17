# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 20:55:16 2022

@author: Lyle
"""

from pymoo.factory import get_performance_indicator
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import random
from tqdm import trange
import tensorflow as tf
import eval_prd
import importlib
importlib.reload(eval_prd)


def Hypervolume_wrapper(hv_ref="auto"):
    def Hypervolume(x_eval, y_eval, x_data, y_data, n_data, scorebars, hv_ref=hv_ref):
        y_eval = np.array(y_eval)
        if scorebars:
            print("Calculating Hypervolume")
        if hv_ref=="auto":
            hv_ref = np.quantile(y_eval, 0.99, axis=0)
        hv = get_performance_indicator("hv", ref_point=hv_ref)
        hvol=hv.do(y_eval)
        return None, hvol
    return Hypervolume


def L2_vectorized(X, Y):
    #Vectorize L2 calculation using x^2+y^2-2xy
    X_sq = tf.reduce_sum(tf.square(X), axis=1)
    Y_sq = tf.reduce_sum(tf.square(Y), axis=1)
    sq = tf.add(tf.expand_dims(X_sq, axis=-1), tf.transpose(Y_sq)) - 2*tf.matmul(X,tf.transpose(Y))
#     print(tf.expand_dims(X_sq, axis=-1), tf.transpose(Y_sq)) - 2*tf.matmul(X,tf.transpose(Y))
    sq = tf.clip_by_value(sq, 0, 1e12, name=None)
    return tf.math.sqrt(sq)

def gen_gen_distance_wrapper(flag, reduction="min"):
    def gen_gen_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag):
        if scorebars:
            print("Calculating Gen-Gen Distance")
        scores = []
        y_eval = StandardScaler().fit_transform(y_eval)
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        res = L2_vectorized(x, x)
        
        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_min(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, tf.reduce_mean(scores)
    return gen_gen_distance


def gen_gen_logdet_wrapper(subset_size=10, sample_times=100000):
    def gen_gen_logdet(x_eval, y_eval, x_data, y_data, n_data, scorebars, subset_size=subset_size, sample_times=sample_times):
        # Average log determinant
        data = x_eval.copy()
        random.seed()
        N = data.shape[0]
        data = data.reshape(N, -1)
        mean_logdet = 0
        num_eval = np.shape(x_eval)[0]
        if scorebars:
            steps_range = trange((num_eval), desc='Calculating Gen-gen Logdet:', leave=True, ascii ="         =")
        else:
            steps_range = range(num_eval)
        for i in range(steps_range):
            ind = np.random.choice(N, size=subset_size, replace=False)
            subset = data[ind]
            D = squareform(pdist(subset, 'euclidean'))
            S = np.exp(-0.5*np.square(D))
            (sign, logdet) = np.linalg.slogdet(S)
            mean_logdet += logdet
        return None, mean_logdet/sample_times
    return gen_gen_logdet


def gen_data_distance_wrapper(flag, reduction="min"):
    def gen_data_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag = flag):
        if scorebars:
            print("Calculating Gen-Data Distance")
        scores = []
        y_eval = StandardScaler().fit_transform(y_eval)
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
            
        num_eval = np.shape(x)[0]
        res = L2_vectorized(x, data)

        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_min(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, tf.reduce_mean(scores)
    return gen_data_distance

def data_gen_distance_wrapper(flag, reduction="min"):
    def data_gen_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag = flag):
        if scorebars:
            print("Calculating Data-Gen Distance")
        scores = []
        y_eval = StandardScaler().fit_transform(y_eval)
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
            
        num_eval = np.shape(x)[0]
        res = L2_vectorized(data, x)

        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_min(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return None, tf.reduce_mean(scores)
    return data_gen_distance

def DTAI_wrapper(ref, p_, a_):
    def DTAI(x_eval, y_eval, x_data, y_data, n_data, scorebars, ref=ref, p_=p_, a_=a_, DTAI_EPS=1e-7):
        p_ = tf.cast(p_, "float32")
        a_ = tf.cast(a_, "float32")
        y_eval = tf.cast(y_eval, "float32")
        if scorebars:
            print("Calculating DTAI")

        #y values must be greater than 0
        y=tf.math.maximum(y_eval, DTAI_EPS)
        x=tf.divide(y, ref)
        case1 = tf.multiply(p_,x)-p_
        p_over_a=tf.divide(p_,a_)
        exponential=tf.exp(tf.multiply(a_, (1-x)))
        case2=tf.multiply(p_over_a, (1-exponential))
        casemask = tf.greater(x, 1)
        casemask = tf.cast(casemask, "float32")
        scores=tf.multiply(case2, casemask) + tf.multiply( case1, (1 - casemask))
        scores=tf.math.reduce_sum(scores, axis=1)         
        smax=tf.math.reduce_sum(p_/a_)
        smin=-tf.math.reduce_sum(p_)

        scores=(scores-smin)/(smax-smin)
        return scores, tf.reduce_mean(scores)
    return DTAI
def minimum_target_ratio_wrapper(ref):
    def minimum_target_ratio(x_eval, y_eval, x_data, y_data, n_data, scorebars, ref=ref):
        if scorebars:
            print("Calculating Minimum Target Ratio")
        y_eval = tf.cast(y_eval, "float32")
        ref = tf.cast(ref, "float32")
        res = tf.divide(y_eval, ref)
        scores = tf.reduce_min(res, axis=1)
        return scores, tf.reduce_mean(scores)
    return minimum_target_ratio

def weighted_target_success_rate_wrapper(ref, p_):
    def weighted_target_success_rate(x_eval, y_eval, x_data, y_data, n_data, scorebars, ref=ref, p_=p_):
        y_eval = tf.cast(y_eval, "float32")
        if scorebars:
            print("Calculating Weighted Target Success Rate")
        num_eval = y_eval[:,0]
        y_eval = tf.cast(y_eval, "float32")
        ref = tf.cast(ref, "float32")
        p_ = tf.cast(p_, "float32")
        res = tf.cast(y_eval>ref, "float32")
        scores = res
        scaled_scores = tf.matmul(scores, tf.expand_dims(p_, -1))/sum(p_)
        return scaled_scores, tf.reduce_mean(scaled_scores)
    return weighted_target_success_rate

def gen_neg_distance_wrapper(reduction = "min"):
    def gen_neg_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating Gen-Neg Distance")
        res = L2_vectorized(x_eval, n_data)
        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_min(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, tf.reduce_mean(scores)
    return gen_neg_distance

def MMD_wrapper(sigma=1, biased=True):
    def MMD(x_eval, y_eval, x_data, y_data, n_data, scorebars, sigma=sigma, biased=biased):
        if scorebars:
            print("Calculating Maximum Mean Discrepancy")
        X = x_eval
        Y = x_data
        gamma = 1 / (2 * sigma**2)
    
        XX = tf.matmul(X, tf.transpose(X))
        XY = tf.matmul(X, tf.transpose(Y))
        YY = tf.matmul(Y, tf.transpose(Y))
    
        X_sqnorms = tf.linalg.diag_part(XX)
        Y_sqnorms = tf.linalg.diag_part(YY)
    
        K_XY = tf.math.exp(-gamma * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    #             -2 * XY + tf.expand_dims(X_sqnorms, 1) + tf.expand_dims(Y_sqnorms, 0)))
    
        K_XX = tf.math.exp(-gamma * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = tf.math.exp(-gamma * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        
        if biased:
            mmd2 = tf.math.reduce_mean(K_XX) + tf.math.reduce_mean(K_YY) - 2 * tf.math.reduce_mean(K_XY)
        else:
            m = K_XX.shape[0]
            n = K_YY.shape[0]
    
            mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
                  + (K_YY.sum() - n) / (n * (n - 1))
                  - 2 * K_XY.mean())
        return None, mmd2.numpy()
    return MMD

def MMD2_wrapper(batch_size = 128, sigma = [1,2,4,8,16], eps=10e-7):
    def MMD2(x_eval, y_eval, x_data, y_data, n_data, scorebars, batch_size = batch_size, sigma = sigma, eps=eps):
        x_eval=tf.cast(x_eval, "float32")
        x_data=tf.cast(x_data, "float32")
        gen_x = tf.data.Dataset.from_tensor_slices(x_eval).batch(batch_size)
        x = tf.data.Dataset.from_tensor_slices(x_data).batch(batch_size)
        print(x)
#         for i in range(
            
            
    #     slim = tf.contrib.slim
    #     x = slim.flatten(x)
    #     gen_x = slim.flatten(gen_x)

        # concatenation of the generated images and images from the dataset
        # first 'N' rows are the generated ones, next 'M' are from the data
        X = tf.concat([gen_x, x],0)

        # dot product between all combinations of rows in 'X'
        XX = tf.matmul(X, tf.transpose(X))

        # dot product of rows with themselves
        X2 = tf.reduce_sum(X * X, 1, keepdims = True)

        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
        exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)

        # scaling constants for each of the rows in 'X'
        s = makeScaleMatrix(batch_size, batch_size)

        # scaling factors of each of the kernel values, corresponding to the
        # exponent values
        S = tf.matmul(s, tf.transpose(s))

        loss1 = 0
        loss2 = 0
        mmd = 0

        # for each bandwidth parameter, compute the MMD value and add them all
        n = batch_size
        n_sq = float(n*n)
        for i in range(len(sigma)):

            # kernel values for each combination of the rows in 'X' 
            kernel_val = tf.exp(1.0 / sigma[i] * exponent)
            mmd += tf.reduce_sum(S * kernel_val)


        return None, tf.sqrt(mmd)
    return MMD2

def recall_wrapper(flag, num_clusters=20, num_angles=1001, num_runs=10, enforce_balance=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating Recall")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=20, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
        return None, recall[num_angles//2+1]
    return calc_prd

def precision_wrapper(flag, num_clusters=20, num_angles=1001, num_runs=10, enforce_balance=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating Precision")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=20, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
        return None, precision[num_angles//2+1]
    return calc_prd

def F1_wrapper(flag, num_clusters=20, num_angles=1001, num_runs=10, enforce_balance=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating F1")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=20, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
        F1 = eval_prd._prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10)
        prd_data = [np.array([precision,recall])]
        return None, F1[num_angles//2+1]
    return calc_prd

def AUC_wrapper(flag, num_clusters=20, num_angles=1001, num_runs=10, enforce_balance=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating AUC")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=20, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
        F1 = eval_prd._prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10)
        prd_data = [np.array([precision,recall])]
        eval_prd.plot(prd_data, labels=None, out_path=None,legend_loc='lower left', dpi=300)
        tot = 0
        for i in range(num_angles):
            tot+=(recall[i]**2+precision[i]**2)/num_angles*np.pi/4
        return None, tot
    return calc_prd

def evaluate_validity(x_fake, validityfunction):
    scores = np.zeros(np.shape(x_fake)[0])
    for i in range(np.shape(x_fake)[0]):
        scores[i] = validityfunction(x_fake[i,0], x_fake[i,1])
    return scores, np.mean(scores)

# def percentileSelect(df, perc, modify=True):
#     df=df.copy()
#     if modify==True:
#         for col in ['Sim 1 Safety Factor', 'Sim 3 Safety Factor']:
#             df[col]=1/df[col]
#         for col in ['Sim 1 Dropout X Disp.', 'Sim 1 Dropout Y Disp.', 'Sim 1 Bottom Bracket X Disp.', 'Sim 1 Bottom Bracket Y Disp.', 'Sim 2 Bottom Bracket Z Disp.', 'Sim 3 Bottom Bracket Y Disp.', 'Sim 3 Bottom Bracket X Rot.', 'Model Mass']:      
#             df[col]=[np.abs(val) for val in df[col].values]
#     return df.quantile(perc)