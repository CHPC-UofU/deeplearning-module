#!/usr/bin/env python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print("get_build_info:")
print(tf.sysconfig.get_build_info())
print("get_compile_flags:")
print(tf.sysconfig.get_compile_flags())
print("get_include:")
print(tf.sysconfig.get_include())
print("get_lib:")
print(tf.sysconfig.get_lib())
print("get_link_flags:")
print(tf.sysconfig.get_link_flags())

with tf.device('/cpu:0'):
    a_c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-cpu')
    b_c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-cpu')
    c_c = tf.matmul(a_c, b_c, name='c-cpu')

with tf.device('/gpu:0'):
    a_g = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-gpu')
    b_g = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-gpu')
    c_g = tf.matmul(a_g, b_g, name='c-gpu')

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c_c))
    print(sess.run(c_g))

print('DONE!')
