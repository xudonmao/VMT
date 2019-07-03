import tensorflow as tf
import pdb
import tensorbayes as tb
import numpy as np
from codebase.args import args
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent

@add_arg_scope
def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope, 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=range(1, len(d.shape)))
    return output

@add_arg_scope
def scale_gradient(x, scale, scope=None, reuse=None):
    with tf.name_scope('scale_grad'):
        output = (1 - scale) * tf.stop_gradient(x) + scale * x
    return output

@add_arg_scope
def noise(x, std, phase, scope=None, reuse=None):
    with tf.name_scope(scope, 'noise'):
        eps = tf.random_normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
    return output

@add_arg_scope
def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name, 'leaky_relu'):
        return tf.maximum(x, a * x)

@add_arg_scope
def basic_accuracy(a, b, scope=None):
    with tf.name_scope(scope, 'basic_acc'):
        a = tf.argmax(a, 1)
        b = tf.argmax(b, 1)
        eq = tf.cast(tf.equal(a, b), 'float32')
        output = tf.reduce_mean(eq)
    return output

@add_arg_scope
def perturb_image(x, p, classifier, pert='vat', scope=None):
    with tf.name_scope(scope, 'perturb_image'):
        eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

        # Predict on randomly perturbed image
        eps_p = classifier(x + eps, phase=True, reuse=True)
        loss = softmax_xent_two(labels=p, logits=eps_p)

        # Based on perturbed image, get direction of greatest error
        eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

        # Use that direction as adversarial perturbation
        eps_adv = normalize_perturbation(eps_adv)
        x_adv = tf.stop_gradient(x + args.radius * eps_adv)

    return x_adv

@add_arg_scope
def vat_loss(x, p, classifier, scope=None):
    with tf.name_scope(scope, 'smoothing_loss'):
        x_adv = perturb_image(x, p, classifier)
        p_adv = classifier(x_adv, phase=True, reuse=True)
        loss = tf.reduce_mean(softmax_xent_two(labels=tf.stop_gradient(p), logits=p_adv))

    return loss

def mix_up(x, y, bs, alpha):

    weight = np.random.beta(alpha, alpha, bs)
    x_weight = weight.reshape(bs, 1, 1, 1)
    y_weight = weight.reshape(bs, 1)

    x1, x2 = x, tf.gather(x, tf.random_shuffle(tf.range(tf.shape(x)[0])))
    y1, y2 = y, tf.gather(y, tf.random_shuffle(tf.range(tf.shape(y)[0])))

    mix_x = x1 * x_weight + x2 * (1 - x_weight)
    mix_y = y1 * y_weight + y2 * (1 - y_weight)

    return mix_x, mix_y

@add_arg_scope
def logit_mix_loss(x, p, classifier, scope=None):
    with tf.name_scope(scope, 'logit_mix_loss'):

        mix_x, mix_y_logit = mix_up(x, p, 64, 1.0)

        mix_x_e = classifier(
                mix_x, phase=True, enc_phase=1,
                trim=args.trim, reuse=True
                )
        mix_x_p = classifier(
                mix_x_e, phase=True, enc_phase=0,
                trim=args.trim, reuse=True
                )
        loss = tf.reduce_mean(softmax_xent_two(labels=tf.stop_gradient(mix_y_logit),
                                                logits=mix_x_p))
    return loss

@add_arg_scope
def softmax_mix_loss(x, p, classifier, scope=None):
    with tf.name_scope(scope, 'softmax_mix_loss'):
        x_adv = perturb_image(x, p, classifier)

        mix_x, mix_y = mix_up(x, tf.nn.softmax(p), 64, 1.0)

        mix_x_e = classifier(
                mix_x, phase=True, enc_phase=1,
                trim=args.trim, reuse=True
                )
        mix_x_p = classifier(
                mix_x_e, phase=True, enc_phase=0,
                trim=args.trim, reuse=True
                )
        loss = tf.reduce_mean(softmax_xent(labels=tf.stop_gradient(mix_y),
                                                logits=mix_x_p))
    return loss

