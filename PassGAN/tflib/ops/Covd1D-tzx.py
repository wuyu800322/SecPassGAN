{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset134 PingFangSC-Regular;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import tflib as lib\
import numpy as np\
import tensorflow as tf\
\
_default_weightnorm = False\
\
\
def enable_default_weightnorm():\
    global _default_weightnorm\
    _default_weightnorm = True\
\
\
def Conv1D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1):\
    """\
    inputs: tensor of shape (batch size, num channels, width)\
    mask_type: one of None, 'a', 'b'\
\
    returns: tensor of shape (batch size, num channels, width)\
    """\
    with tf.name_scope(name) as scope:\
\
        if mask_type is not None:\
            mask_type, mask_n_channels = mask_type\
\
            mask = np.ones(\
                (filter_size, input_dim, output_dim),\
                dtype='float32'\
            )\
            center = filter_size // 2\
\
            # Mask out future locations\
            mask[center+1:, :, :] = 0.\
\
            # Mask out future channels\
            for i in range(mask_n_channels):\
                for j in range(mask_n_channels):\
                    if (mask_type == 'a' and i >= j) or (mask_type == 'b' and i > j):\
                        mask[\
                            center,\
                            i::mask_n_channels,\
                            j::mask_n_channels\
                        ] = 0.\
\
        def uniform(stdev, size):\
            return np.random.uniform(\
                low=-stdev * np.sqrt(3),\
                high=stdev * np.sqrt(3),\
                size=size\
            ).astype('float32')\
\
        fan_in = input_dim * filter_size\
        fan_out = output_dim * filter_size / stride\
\
        if mask_type is not None:  # only approximately correct\
            fan_in /= 2.\
            fan_out /= 2.\
\
        if he_init:\
            filters_stdev = np.sqrt(4./(fan_in+fan_out))\
        else:  # Normalized init (Glorot & Bengio)\
            filters_stdev = np.sqrt(2./(fan_in+fan_out))\
\
        filter_values = uniform(\
            filters_stdev,\
            (filter_size, input_dim, output_dim)\
        )\
        filter_values *= gain\
\
        filters = lib.param(name+'.Filters', filter_values)\
\
        if weightnorm is None:\
            weightnorm = _default_weightnorm\
        if weightnorm:\
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0, 1)))\
            target_norms = lib.param(\
                name + '.g',\
                norm_values\
            )\
            with tf.name_scope('weightnorm') as scope:\
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), axis=(0, 1)))\
                filters = filters * (target_norms / norms)\
\
        if mask_type is not None:\
            with tf.name_scope('filter_mask'):\
                filters = filters * mask\
\
        # ===== 
\f1 \'d0\'de\'b8\'c4\'ca\'e4\'c8\'eb\'ca\'fd\'be\'dd\'b8\'f1\'ca\'bd
\f0  =====\
        # 
\f1 \'d4\'ad\'ca\'bc\'b4\'fa\'c2\'eb\'a3\'a8\'b1\'bb\'d7\'a2\'ca\'cd\'b5\'f4\'a3\'a9
\f0 \
        # result = tf.nn.conv1d(  # 
\f1 \'d4\'ad\'ca\'bc\'b4\'fa\'c2\'eb\'a3\'ac\'ca\'b9\'d3\'c3
\f0  NCW 
\f1 \'b8\'f1\'ca\'bd\'a3\'ac\'b2\'bb\'b1\'bb
\f0  CPU 
\f1 \'d6\'a7\'b3\'d6
\f0 \
        #     input=inputs,\
        #     filters=filters,\
        #     stride=stride,\
        #     padding='SAME',\
        #     data_format='NCW'  # 
\f1 \'b2\'bb\'d6\'a7\'b3\'d6
\f0 \
        # )\
\
        # 
\f1 \'d0\'c2\'b4\'fa\'c2\'eb\'a3\'ba\'bd\'ab\'ca\'e4\'c8\'eb\'b4\'d3
\f0  NCW 
\f1 \'d7\'aa\'bb\'bb\'ce\'aa
\f0  NWC\
        inputs = tf.transpose(inputs, [0, 2, 1])  # NCW -> NWC 
\f1 \'b8\'f1\'ca\'bd
\f0 \
\
        # 
\f1 \'d0\'c2\'b4\'fa\'c2\'eb\'a3\'ba\'d6\'b4\'d0\'d0\'be\'ed\'bb\'fd\'b2\'d9\'d7\'f7
\f0 \
        result = tf.nn.conv1d(\
            input=inputs,\
            filters=filters,\
            stride=stride,\
            padding='SAME'  # 
\f1 \'c4\'ac\'c8\'cf\'d6\'a7\'b3\'d6
\f0  NWC 
\f1 \'b8\'f1\'ca\'bd
\f0 \
        )\
\
        # 
\f1 \'d0\'c2\'b4\'fa\'c2\'eb\'a3\'ba\'bd\'ab\'ca\'e4\'b3\'f6\'b4\'d3
\f0  NWC 
\f1 \'d7\'aa\'bb\'bb\'bb\'d8
\f0  NCW 
\f1 \'b8\'f1\'ca\'bd
\f0 \
        result = tf.transpose(result, [0, 2, 1])  # NWC -> NCW 
\f1 \'b8\'f1\'ca\'bd
\f0 \
\
        # ===== 
\f1 \'d0\'de\'b8\'c4
\f0  Bias 
\f1 \'b5\'c4\'cc\'ed\'bc\'d3\'b2\'bf\'b7\'d6
\f0  =====\
        if biases:\
            _biases = lib.param(\
                name+'.Biases',\
                np.zeros([output_dim], dtype='float32')\
            )\
\
            # 
\f1 \'d4\'ad\'ca\'bc\'b4\'fa\'c2\'eb\'a3\'a8\'b1\'bb\'d7\'a2\'ca\'cd\'b5\'f4\'a3\'a9
\f0 \
            # result = tf.expand_dims(result, 3)\
            # result = tf.nn.bias_add(result, _biases, data_format='NCW')  # 
\f1 \'b2\'bb\'d6\'a7\'b3\'d6
\f0 \
            # result = tf.squeeze(result)\
\
            # 
\f1 \'d0\'c2\'b4\'fa\'c2\'eb\'a3\'ba\'d6\'b1\'bd\'d3\'cc\'ed\'bc\'d3
\f0  bias
\f1 \'a3\'ac\'d2\'f2\'ce\'aa\'ca\'fd\'be\'dd\'d2\'d1\'d7\'aa\'bb\'bb\'ce\'aa
\f0  NWC 
\f1 \'b8\'f1\'ca\'bd
\f0 \
            result = tf.nn.bias_add(result, _biases)\
\
        return result\
}