import importlib
import pdb
import tensorflow as tf
import tensorbayes as tb
import numpy as np
from extra_layers import basic_accuracy, vat_loss, logit_mix_loss, softmax_mix_loss
from codebase.args import args
from pprint import pprint
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorbayes.layers import placeholder, constant
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_xent
nn = importlib.import_module('codebase.models.nns.{}'.format(args.nn))

def dirtt(param):
    dw  = param.dw    
    cw  = 1           
    svw = param.svw   
    tvw = param.tvw   
    tcw = param.tcw   
    smw = param.smw
    tmw = param.tmw

    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=tb.growth_config()),
        src_x = placeholder((None, 32, 32, 3)),
        src_y = placeholder((None, args.Y)),
        trg_x = placeholder((None, 32, 32, 3)),
        trg_y = placeholder((None, args.Y)),
        test_x = placeholder((None, 32, 32, 3)),
        test_y = placeholder((None, args.Y)),
    ))
    # Supervised and conditional entropy minimization
    src_e = nn.classifier(T.src_x, phase=True, enc_phase=1, trim=args.trim)
    trg_e = nn.classifier(T.trg_x, phase=True, enc_phase=1, trim=args.trim, reuse=True, internal_update=True)
    src_p = nn.classifier(src_e, phase=True, enc_phase=0, trim=args.trim)
    trg_p = nn.classifier(trg_e, phase=True, enc_phase=0, trim=args.trim, reuse=True, internal_update=True)

    loss_src_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_p))
    loss_trg_cent = tf.reduce_mean(softmax_xent_two(labels=trg_p, logits=trg_p)) if tcw > 0 else tf.constant(0, 'float32')


    # Domain confusion
    if param.dw > 0 and args.dirt == 0:
        real_logit = nn.feature_discriminator(src_e, phase=True)
        fake_logit = nn.feature_discriminator(trg_e, phase=True, reuse=True)

        loss_disc = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
        loss_domain = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.zeros_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))

    else:
        loss_disc = constant(0)
        loss_domain = constant(0)

    # Virtual adversarial training (turn off src in non-VADA phase)
    loss_src_vat = vat_loss(T.src_x, src_p, nn.classifier) if svw > 0 and args.dirt == 0 else tf.constant(0, 'float32') 
    loss_trg_vat = vat_loss(T.trg_x, trg_p, nn.classifier) if tvw > 0 else tf.constant(0, 'float32')

    loss_src_mix_adv = logit_mix_loss(T.src_x, src_p, 
            nn.classifier) if smw > 0 and args.dirt == 0 else tf.constant(0, 'float32')
    loss_trg_mix_adv = logit_mix_loss(T.trg_x, trg_p, 
            nn.classifier) if tmw > 0 else tf.constant(0, 'float32')

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    var_class = tf.get_collection('trainable_variables', 'class/')
    ema_op = ema.apply(var_class)
    ema_p = nn.classifier(T.test_x, phase=False, reuse=True, getter=tb.tfutils.get_getter(ema))

    # Teacher model (a back-up of EMA model)
    teacher_p = nn.classifier(T.test_x, phase=False, scope='teacher')
    var_main = tf.get_collection('variables', 'class/(?!.*ExponentialMovingAverage:0)')
    var_teacher = tf.get_collection('variables', 'teacher/(?!.*ExponentialMovingAverage:0)')
    teacher_assign_ops = []
    for t, m in zip(var_teacher, var_main):
        ave = ema.average(m)
        ave = ave if ave else m
        teacher_assign_ops += [tf.assign(t, ave)]
    update_teacher = tf.group(*teacher_assign_ops)
    teacher = tb.function(T.sess, [T.test_x], tf.nn.softmax(teacher_p))

    # Accuracies
    src_acc = basic_accuracy(T.src_y, src_p)
    trg_acc = basic_accuracy(T.trg_y, trg_p)
    ema_acc = basic_accuracy(T.test_y, ema_p)
    fn_ema_acc = tb.function(T.sess, [T.test_x, T.test_y], ema_acc)


    dw = constant(dw) if args.dirt == 0 else constant(0)    
    cw = constant(cw) if args.dirt == 0 else constant(args.bw)      
    svw = constant(svw) if args.dirt == 0 else constant(0)     
    tvw = constant(tvw)   
    tcw = constant(tcw)   
    smw = constant(smw) if args.dirt == 0 else constant(0)  
    tmw = constant(tmw)   
    loss_main = (dw * loss_domain +
                 cw * loss_src_class +
                 smw * loss_src_mix_adv +
                 tmw * loss_trg_mix_adv +
                 svw * loss_src_vat +
                 tcw * loss_trg_cent +
                 tvw * loss_trg_vat
                 )
    var_main = tf.get_collection('trainable_variables', 'class')
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)
    train_main = tf.group(train_main, ema_op)

    if param.dw > 0 and args.dirt == 0:
        var_disc = tf.get_collection('trainable_variables', 'disc')
        train_disc = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_disc, var_list=var_disc)
    else:
        train_disc = constant(0)

    # Summarizations
    #summary_disc = [tf.summary.scalar('domain/loss_disc', loss_disc),]
    #summary_main = [tf.summary.scalar('domain/loss_domain', loss_domain),
    #                tf.summary.scalar('class/loss_src_class', loss_src_class),
    #                tf.summary.scalar('class/loss_trg_cent', loss_trg_cent),
    #                tf.summary.scalar('lipschitz/loss_trg_vat', loss_trg_vat),
    #                tf.summary.scalar('lipschitz/loss_src_vat', loss_src_vat),
    #                tf.summary.scalar('hyper/dw', dw),
    #                tf.summary.scalar('hyper/cw', cw),
    #                tf.summary.scalar('hyper/sw', sw),
    #                tf.summary.scalar('hyper/tw', tw),
    #                tf.summary.scalar('acc/src_acc', src_acc),
    #                tf.summary.scalar('acc/trg_acc', trg_acc)]

    # Merge summaries
    #summary_disc = tf.summary.merge(summary_disc)
    #summary_main = tf.summary.merge(summary_main)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('disc'), loss_disc,
                   c('domain'), loss_domain,
                   c('class'), loss_src_class,
                   c('cent'), loss_trg_cent,
                   c('trg_vat'), loss_trg_vat,
                   c('src_vat'), loss_src_vat,
                   c('src_mix_adv'), loss_src_mix_adv,
                   c('trg_mix_adv'), loss_trg_mix_adv,
                   c('src'), src_acc,
                   c('trg'), trg_acc,
                   c('svw'), svw,
                   c('tvw'), tvw,
                   c('tcw'), tcw,
                   c('smw'), smw,
                   c('tmw'), tmw
                   ]
    T.ops_disc = train_disc
    T.ops_main = train_main
    T.fn_ema_acc = fn_ema_acc
    T.teacher = teacher
    T.update_teacher = update_teacher

    return T
