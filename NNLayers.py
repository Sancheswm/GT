import tensorflow as tf
# No longer using tf.contrib.layers
import numpy as np

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
    global paramId
    paramId += 1
    return paramId

def setIta(ITA):
    global ita  # Declare ita as global
    ita = ITA

def setBiasDefault(val):
    global biasDefault
    biasDefault = val

def getParam(name):
    return params[name]

def addReg(name, param):
    global regParams
    if name not in regParams:
        regParams[name] = param
    else:
        print('ERROR: Parameter already exists')

def addParam(name, param):
    global params
    if name not in params:
        params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
    name = 'defaultParamName%d' % getParamId()
    return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
    global params
    global regParams
    assert name not in params, 'name %s already exists' % name

    if initializer == 'xavier':
        ret = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=shape, dtype=dtype), name=name, trainable=trainable, dtype=dtype)
    elif initializer == 'trunc_normal':
        ret = tf.Variable(tf.random.truncated_normal(shape=shape, mean=0.0, stddev=0.03, dtype=dtype), name=name, trainable=trainable, dtype=dtype)
    elif initializer == 'zeros':
        ret = tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=name, trainable=trainable, dtype=dtype)
    elif initializer == 'ones':
        ret = tf.Variable(tf.ones(shape=shape, dtype=dtype), name=name, trainable=trainable, dtype=dtype)
    elif callable(initializer):  # Check if it's a callable initializer
        ret = tf.Variable(initializer(shape=shape, dtype=dtype), name=name, trainable=trainable, dtype=dtype)
    else:
        raise ValueError('ERROR: Unrecognized initializer')  # Raise ValueError

    params[name] = ret
    if reg:
        regParams[name] = ret
    return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
    global params
    global regParams
    if name in params:
        assert reuse, 'Reusing Param %s Not Specified' % name
        if reg and name not in regParams:
            regParams[name] = params[name]
        return params[name]
    return defineParam(name, shape, dtype, reg, initializer, trainable)


def BN(inp, name=None):
    global ita
    dim = inp.shape[1]  # Use .shape for eager tensors
    name = 'defaultParamName%d' % getParamId()
    scale = tf.Variable(tf.ones([dim]), name=name + "_scale")  # Add a name
    shift = tf.Variable(tf.zeros([dim]), name=name + "_shift")  # Add a name
    fcMean, fcVar = tf.nn.moments(inp, axes=[0])
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    emaApplyOp = ema.apply([fcMean, fcVar])
    with tf.control_dependencies([emaApplyOp]):
        mean = tf.identity(fcMean)
        var = tf.identity(fcVar)
    ret = tf.nn.batch_normalization(inp, mean, var, shift, scale, 1e-8)
    return ret


def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False):
    inDim = inp.shape[1]  # Use .shape for eager tensors
    temName = name if name != None else 'defaultParamName%d' % getParamId()
    W = getOrDefineParam(temName + "_W", [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse) # Added "_W" suffix
    if dropout != None:
        ret = tf.nn.dropout(inp, rate=dropout) @ W
    else:
        ret = inp @ W

    if useBias:
        ret = Bias(ret, name=name, reuse=reuse)
    if useBN:
        ret = BN(ret)
    if activation != None:
        ret = Activate(ret, activation)
    return ret

def Bias(data, name=None, reg=False, reuse=False):
    inDim = data.shape[-1]  # Use .shape for eager tensors
    temName = name if name != None else 'defaultParamName%d' % getParamId()
    temBiasName = temName + 'Bias'
    bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer='zeros', reuse=reuse)
    if reg:
        regParams[temBiasName] = bias
    return data + bias

def ActivateHelp(data, method):
    if method == 'relu':
        ret = tf.nn.relu(data)
    elif method == 'sigmoid':
        ret = tf.nn.sigmoid(data)
    elif method == 'tanh':
        ret = tf.nn.tanh(data)
    elif method == 'softmax':
        ret = tf.nn.softmax(data, axis=-1)
    elif method == 'leakyRelu':
        ret = tf.maximum(leaky * data, data)
    elif method == 'twoWayLeakyRelu6':
        temMask = tf.cast(tf.greater(data, 6.0), tf.float32) # tf.to_float is deprecated
        ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
    elif method == '-1relu':
        ret = tf.maximum(-1.0, data)
    elif method == 'relu6':
        ret = tf.maximum(0.0, tf.minimum(6.0, data))
    elif method == 'relu3':
        ret = tf.maximum(0.0, tf.minimum(3.0, data))
    else:
        raise ValueError('Error Activation Function')  # Raise ValueError
    return ret

def Activate(data, method, useBN=False):
    global leaky
    if useBN:
        ret = BN(data)
    else:
        ret = data
    ret = ActivateHelp(ret, method)
    return ret


def Regularize(names=None, method='L2'):
    ret = 0
    if method == 'L1':
        if names != None:
            for name in names:
                ret += tf.reduce_sum(tf.abs(getParam(name)))
        else:
            for name in regParams:
                ret += tf.reduce_sum(tf.abs(regParams[name]))
    elif method == 'L2':
        if names != None:
            for name in names:
                ret += tf.reduce_sum(tf.square(getParam(name)))
        else:
            for name in regParams:
                ret += tf.reduce_sum(tf.square(regParams[name]))
    return ret

def Dropout(data, rate):
    if rate is None:  # Corrected condition
        return data
    else:
        return tf.nn.dropout(data, rate=rate)


def selfAttention(localReps, number, inpDim, numHeads):
    Q = defineRandomNameParam([inpDim, inpDim], reg=True)
    K = defineRandomNameParam([inpDim, inpDim], reg=True)
    V = defineRandomNameParam([inpDim, inpDim], reg=True)
    rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
    q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim // numHeads])
    k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim // numHeads])
    v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim // numHeads])

    # Improved attention calculation (more stable)
    logits = tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(float(inpDim / numHeads)) # Cast to float for tf.sqrt
    att = tf.nn.softmax(logits, axis=2)  # Softmax over the number dimension

    attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
    rets = [None] * number

    for i in range(number):
        tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
        rets[i] = tem1 + localReps[i]
    return rets


def lightSelfAttention(localReps, number, inpDim, numHeads):
    Q = defineRandomNameParam([inpDim, inpDim], reg=True)
    rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
    tem = rspReps @ Q
    q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim // numHeads])
    k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim // numHeads])
    v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim // numHeads])

    # Improved attention calculation (more stable)
    logits = tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(float(inpDim / numHeads)) # Cast to float for tf.sqrt
    att = tf.nn.softmax(logits, axis=2)  # Softmax over the number dimension

    attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
    rets = [None] * number

    for i in range(number):
        tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
        rets[i] = tem1 + localReps[i]
    return rets  # , tf.squeeze(att)  If you need the attention weights, uncomment this line
