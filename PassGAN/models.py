import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import numpy as np
import hashlib

def ResBlock(name, inputs, dim):
    output = inputs
    output = tf.nn.relu(output)
    # print(name+'.1', dim, dim, 5, output)
    output = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 5, output)
    return inputs + (0.3*output)


def password_to_noise(input_password, noise_dim=128, hash_method='md5'):
    """
    将单个密码转换为固定长度的噪声向量。
    :param input_password: 输入的单个密码（字符串）。
    :param noise_dim: 噪声向量的长度。
    :param hash_method: 哈希方法（默认使用 SHA-256）。
    :return: 噪声向量（长度为 noise_dim）。
    """
    # 支持多种哈希算法
    if hash_method == 'sha256':
        hashed = hashlib.sha256(input_password.encode('utf-8')).hexdigest()
    elif hash_method == 'md5':
        hashed = hashlib.md5(input_password.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash method: {hash_method}")
    
 
    # 将哈希值转为噪声数组，范围在 [0, 1]
    noise = np.array([int(hashed[i:i+2], 16) / 255.0 for i in range(0, len(hashed), 2)])
    # 如果噪声长度不足，随机重复已有噪声进行填充
    if len(noise) < noise_dim:
        # 计算需要补充的长度
        pad_length = noise_dim - len(noise)
        # 随机从已有噪声中选择值进行填充
        padding = np.random.choice(noise, size=pad_length, replace=True)
        noise = np.concatenate([noise, padding])
    elif len(noise) > noise_dim:
        # 如果长度过长，则截取前 noise_dim 长度
        noise = noise[:noise_dim]

    return noise

def passwords_to_noise_array(passwords, noise_dim=128):
    """
    将密码数组转化为噪声数组。
    :param passwords: 密码数组（列表或数组）。
    :param noise_dim: 噪声向量的长度。
    :return: 噪声数组，形状为 [len(passwords), noise_dim]。
    """
    # 对每个密码调用 password_to_noise
    return np.array([password_to_noise(pwd, noise_dim) for pwd in passwords])

def Generator(n_samples, seq_len, layer_dim, output_dim, input_passwords=None):
    """
    修改后的生成器模型，支持通过密码数组调整噪声向量，自动截取或补充密码数组。
    :param n_samples: 生成的样本数量
    :param seq_len: 样本的序列长度
    :param layer_dim: 隐藏层维度
    :param output_dim: 输出维度（字符映射表大小）
    :param input_passwords: 输入的密码数组，用于生成噪声数组
    :return: 生成器输出
    """
    # 如果提供了密码数组，处理噪声
    if input_passwords:
        # 如果密码数组长度少于 n_samples，循环补充
        if len(input_passwords) < n_samples:
            input_passwords = (input_passwords * ((n_samples // len(input_passwords)) + 1))[:n_samples]
        # 如果密码数组长度大于 n_samples，截取前 n_samples 个
        elif len(input_passwords) > n_samples:
            input_passwords = input_passwords[:n_samples]

        # 转换为噪声数组
        password_noise_array = passwords_to_noise_array(input_passwords, noise_dim=128)
        print('\npassword_noise_array:\n')
        print(password_noise_array)
        noise = tf.convert_to_tensor(password_noise_array, dtype=tf.float32)
    else:
        # 如果没有提供密码数组，生成随机噪声
        noise = make_noise(shape=[n_samples, 128])  # 随机噪声

    # 生成器网络
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_len * layer_dim, noise)
    output = tf.reshape(output, [-1, layer_dim, seq_len])
    output = ResBlock('Generator.1', output, layer_dim)
    output = ResBlock('Generator.2', output, layer_dim)
    output = ResBlock('Generator.3', output, layer_dim)
    output = ResBlock('Generator.4', output, layer_dim)
    output = ResBlock('Generator.5', output, layer_dim)
    output = lib.ops.conv1d.Conv1D('Generator.Output', layer_dim, output_dim, 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output, output_dim)
    return output

def Discriminator(inputs, seq_len, layer_dim, input_dim):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', input_dim, layer_dim, 1, output)
    output = ResBlock('Discriminator.1', output, layer_dim)
    output = ResBlock('Discriminator.2', output, layer_dim)
    output = ResBlock('Discriminator.3', output, layer_dim)
    output = ResBlock('Discriminator.4', output, layer_dim)
    output = ResBlock('Discriminator.5', output, layer_dim)
    output = tf.reshape(output, [-1, seq_len * layer_dim])
    output = lib.ops.linear.Linear('Discriminator.Output', seq_len * layer_dim, 1, output)
    return output

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random.normal(shape)
