#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 忽略所有警告信息
import time
import pickle
import argparse
import base64, zlib
from pathlib import Path
import random
import string
import csv
import string
#####

import sys
sys.path.append(os.getcwd())  # 将当前工作目录添加到 Python 模块搜索路径中

# 加载 CUDA 动态链接库（用于加速计算）
import ctypes
import ctypes.util
name = ctypes.util.find_library('cudart64_110.dll')  # 查找 CUDA Runtime 动态库
lib = ctypes.cdll.LoadLibrary(name)

# 加载其他 CUDA 库，如有需要可以取消注释并加载其他库
# name = ctypes.util.find_library('cublas64_11.dll')
# lib = ctypes.cdll.LoadLibrary(name)

name = ctypes.util.find_library('cufft64_10.dll')  # 查找 CUDA FFT 库
lib = ctypes.cdll.LoadLibrary(name)

name = ctypes.util.find_library('curand64_10.dll')  # 查找 CUDA 随机数库
lib = ctypes.cdll.LoadLibrary(name)

name = ctypes.util.find_library('cusparse64_11.dll')  # 查找 CUDA 稀疏矩阵库
lib = ctypes.cdll.LoadLibrary(name)

# 加载深度学习库，如需要可以加载 cudnn
# name = ctypes.util.find_library('cudnn64_8.dll')
# lib = ctypes.cdll.LoadLibrary(name)

# 打印启动横幅
def print_banner():
    # 使用 base64 和 gzip 压缩的横幅信息
    encoded_data = "H4sIAAAAAAAAA3VRWw6AMAj73yn6qYkJFzKZB+Hw0uJ8oJLYYSmPMWBYTwMq8yLbCKKKUSy0g/wK31roJzOOkIWPVbFrkG5IBJwSj89a5oYXAaVtkh3AI9AlhpNo6jhqMW0i4YT5LNMlVh9oKlNjDrV0U65gTZfFdbhkjZf5X25ZyXXvuqnnK8jZAcxviKq1AQAA"
    banner = zlib.decompress(base64.b64decode(encoded_data), 16 + zlib.MAX_WBITS).decode('utf-8')
    print(banner)

print_banner()

print("\nLoading TensorFlow...\n")  # 提示正在加载 TensorFlow

# 导入必要的工具包
import utils
import models
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 设定日志等级，仅显示错误信息
import numpy as np
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
print("")

import tensorflow.compat.v1 as tf  # 使用兼容模式的 TensorFlow 1.x
tf.disable_v2_behavior()  # 禁用 TensorFlow 2.x 的行为

import tflib.plot  # 用于绘制训练过程中的图表

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = infile.readlines()
        writer = csv.writer(outfile)

        # 写入CSV标题
        writer.writerow(["Password", "Score"])

        # 逐行写入内容到CSV
        for line in reader:
            password, score = line.strip().split('\t')
            writer.writerow([password, score])

    print(f"Conversion completed. Results saved to {csv_file}")


######### 生成样本 #########
def generator_run(args):

    # 确保 input_dir 是 Path 对象
    input_dir = Path(args.input_dir)

    # 加载字符映射文件
    with open(input_dir / 'charmap.pickle', 'rb') as f:
        charmap = pickle.load(f)

    with open(input_dir / 'inv_charmap.pickle', 'rb') as f:
        inv_charmap = pickle.load(f)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        passwords = f.read().splitlines()
        
    # 构建生成器模型
    fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap),passwords)

    with tf.compat.v1.Session() as session:

        # 生成密码样本
        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(inv_charmap[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            return decoded_samples

        # 当密码长度不足时进行补充
        def pad_sample(sample):
            # 从 inv_charmap 中过滤出仅包含英文字母和数字的字符
            alnum_chars = [char for char in inv_charmap if char in string.ascii_letters + string.digits]
            target_length = random.randint((1 << 3) + (1 << 1) + 1 , 12)  

            # 如果 alnum_chars 为空，提示并返回原样
            if not alnum_chars:
                print("Warning: No valid alphanumeric characters found in inv_charmap.")
                return sample

            symbol_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/`~"       # 符号字符集
            uppercase_chars = string.ascii_uppercase             # 大写字母字符集

            # 补全逻辑
            while len(sample) < target_length:
                if random.random() < 0.5:  # 50% 概率选择符号或大写字母
                    sample += random.choice(symbol_chars + uppercase_chars)
                else:
                    sample += random.choice(alnum_chars)  # 保留原始字符集合中的字符

                
            return sample[:target_length]  # 截取至目标长度
        # 保存生成的样本
        def save(samples):
            with open(args.output, 'a') as f:
                for s in samples:
                    padded_sample = pad_sample("".join(s).replace('`', ''))  # 清除 ` 字符并补充长度
                    f.write(padded_sample + "\n")
                   # txt_to_csv(args.output, args.output.replace('.txt', '.csv'))  # 将输出的txt转为csv

        saver = tf.compat.v1.train.Saver()  # 恢复训练好的模型
        saver.restore(session, args.checkpoint)

        samples = []
        then = time.time()
        start = time.time()
        for i in range(int(args.num_samples / args.batch_size)):

            samples.extend(generate_samples())

            # 每 1000 批次保存一次生成的样本
            if i % 1000 == 0 and i > 0:
                save(samples)
                samples = []  # 清空样本缓冲
                print(f'wrote {1000 * args.batch_size} samples to {args.output} in {time.time() - then:.2f} seconds. {i * args.batch_size} total.')
                then = time.time()

        save(samples)
        print(f'finished in {time.time() - start:.2f} seconds')  # 输出完成时间

##################################

######### 生成样本 #########
def sample_run(args):

    # 确保 input_dir 是 Path 对象
    input_dir = Path(args.input_dir)

    # 加载字符映射文件
    with open(input_dir / 'charmap.pickle', 'rb') as f:
        charmap = pickle.load(f)

    with open(input_dir / 'inv_charmap.pickle', 'rb') as f:
        inv_charmap = pickle.load(f)

    # 构建生成器模型
    fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))

    with tf.compat.v1.Session() as session:

        # 生成密码样本
        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(inv_charmap[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            return decoded_samples

        # 当密码长度不足时进行补充
        def pad_sample(sample):
            target_length = random.randint(10, 12)  # 随机选择最终长度为10-12

            # 如果原始密码已经满足长度要求，直接返回截取后的结果
            if len(sample) >= target_length:
                return sample[:target_length]

            # 如果密码长度不足，重复密码的字符进行补充
            while len(sample) < target_length:
                sample += sample  # 重复密码字符
            return sample[:target_length]  # 截取至目标长度
        
        # 保存生成的样本
        def save(samples):
            with open(args.output, 'a') as f:
                for s in samples:
                    padded_sample = pad_sample("".join(s).replace('`', ''))  # 清除 ` 字符并补充长度
                    f.write(padded_sample + "\n")
                   # txt_to_csv(args.output, args.output.replace('.txt', '.csv'))  # 将输出的txt转为csv

        saver = tf.compat.v1.train.Saver()  # 恢复训练好的模型
        saver.restore(session, args.checkpoint)

        samples = []
        then = time.time()
        start = time.time()
        for i in range(int(args.num_samples / args.batch_size)):

            samples.extend(generate_samples())

            # 每 1000 批次保存一次生成的样本
            if i % 1000 == 0 and i > 0:
                save(samples)
                samples = []  # 清空样本缓冲
                print(f'wrote {1000 * args.batch_size} samples to {args.output} in {time.time() - then:.2f} seconds. {i * args.batch_size} total.')
                then = time.time()

        save(samples)
        print(f'finished in {time.time() - start:.2f} seconds')  # 输出完成时间

###################################

######### 训练模型 #########

def train_run(args):

    # 加载数据集，并生成字符映射
    lines, charmap, inv_charmap = utils.load_dataset(
        path=args.training_data,
        max_length=args.seq_length,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)  # 创建输出目录
    
    # 创建检查点和样本目录
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'samples').mkdir(exist_ok=True)

    # 保存字符映射，避免编码错误
    with open((Path(args.output_dir) / 'charmap.pickle'), 'wb') as f:
        pickle.dump(charmap, f)

    with open((Path(args.output_dir) / 'inv_charmap.pickle'), 'wb') as f:
        pickle.dump(inv_charmap, f)

    # 定义占位符和模型
    real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

    fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
    fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

    disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))
    disc_fake = models.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap))

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)  # 判别器损失
    gen_cost = -tf.reduce_mean(disc_fake)  # 生成器损失

    # WGAN 梯度惩罚
    alpha = tf.random_uniform(
        shape=[args.batch_size,1,1],
        minval=0.,
        maxval=1.
    )

    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(models.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap)), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += args.lamb * gradient_penalty

    # 优化器
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

    # 数据集生成器
    def inf_train_gen():
        while True:
            np.random.shuffle(lines)
            for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
                yield np.array(
                    [[charmap[c] for c in l] for l in lines[i:i+args.batch_size]],
                    dtype='int32'
                )

    # 监控 JS 散度
    true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[10*args.batch_size:], tokenize=False) for i in range(4)]
    validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[:10*args.batch_size], tokenize=False) for i in range(4)]
    for i in range(4):
        print(f"validation set JSD for n={i+1}: {true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])}")
    true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]
    
    # 开始训练
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        # 生成样本的函数
        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(inv_charmap[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            return decoded_samples

        gen = inf_train_gen()

        for iteration in range(args.iters):
            start_time = time.time()

            # 训练生成器
            if iteration > 0:
                _ = session.run(gen_train_op)

            # 训练判别器
            for i in range(args.critic_iters):
                _data = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete:_data}
                )

            lib.plot.output_dir = args.output_dir
            lib.plot.plot('time', time.time() - start_time)
            lib.plot.plot('train disc cost', _disc_cost)

            # 每 100 次迭代保存一次样本
            if iteration % 100 == 0 and iteration > 0:
                samples = []
                for i in range(10):
                    samples.extend(generate_samples())

                for i in range(4):
                    lm = utils.NgramLanguageModel(i+1, samples, tokenize=False)
                    lib.plot.plot(f'js{i+1}', lm.js_with(true_char_ngram_lms[i]))

                with open((Path(args.output_dir) / 'samples' / f'samples_{iteration}.txt'), 'w') as f:
                    for s in samples:
                        s = "".join(s)
                        f.write(s + "\n")

            # 保存检查点
            if iteration % args.save_every == 0 and iteration > 0:
                model_saver = tf.train.Saver()
                save_path = str(Path(args.output_dir) / 'checkpoints' / f'checkpoint_{iteration}.ckpt')
                model_saver.save(session, save_path)
            lib.plot.tick()

###################################



######### Password Strength Evaluation #########

def evaluate_run(args):
    # Ensure input_dir is a Path object
    input_dir = Path(args.input_dir)

    # Load character mapping files 映射关系
    with open(input_dir / 'charmap.pickle', 'rb') as f:
        charmap = pickle.load(f)

    with open(input_dir / 'inv_charmap.pickle', 'rb') as f:
        inv_charmap = pickle.load(f)

    # Placeholder for real password inputs,
    real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

    # Build the discriminator model
    disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))

    with tf.compat.v1.Session() as session:
        saver = tf.compat.v1.train.Saver()  # Restore the trained model
        saver.restore(session, args.checkpoint)

        # Load passwords from the input file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            passwords = f.read().splitlines()

        # Preprocess passwords without truncating
        passwords_original = passwords  # 保留原始密码列表
        print(passwords)
        passwords = [[charmap.get(c, charmap['`']) for c in p] for p in passwords]  # Convert characters to indices

        # Pad passwords to seq_length for model input, but keep the original length for output
        passwords_padded = []
        for p in passwords:
            if len(p) < args.seq_length:
                p = [charmap['`']] * (args.seq_length - len(p)) + p  # Pad at the start to reach seq_length
            passwords_padded.append(p[:args.seq_length])  # Ensure each entry matches seq_length for input

        # Evaluate in batches
        scores = []
        for i in range(0, len(passwords_padded), args.batch_size):
            batch = passwords_padded[i:i+args.batch_size]  # 从密码列表中获取当前批次
            batch_size_actual = len(batch)  # 当前批次的实际大小
            if len(batch) < args.batch_size:
                # Pad batch to batch_size
                batch += [[charmap['`']] * args.seq_length] * (args.batch_size - len(batch))
            feed_dict = {real_inputs_discrete: batch}
            disc_scores = session.run(disc_real, feed_dict=feed_dict)  # 运行判别器模型，得到密码的得分
            scores.extend(disc_scores[:batch_size_actual])  # 只保留当前批次中实际密码的得分

        # Save evaluation results with original passwords
        with open(args.output, 'w', encoding='utf-8') as f:
            for pwd_original, score in zip(passwords_original, scores):  # 使用原始密码
                f.write(f"{pwd_original}\t{score}\n")  # 将密码和对应得分写入文件，每行格式为 '密码\t得分'

        # txt_to_csv(args.output, args.output.replace('.txt', '.csv'))  # 将输出的txt转为csv
        print(f"Evaluation completed. Results saved to {args.output}")

# 显示帮助信息
def help():
   print("A Deep Learning Approach for Password Guessing.\n")
   print("List of arguments:\n")
   print("-h, --help              show this help message and exit")
   print("sample                  use the pretrained model to generate passwords")
   print("train                   train a model on a large dataset (can take several hours on a GTX 1080)")
   print("")
   print("Usage Examples:")
   print("passgan sample --input-dir pretrained --checkpoint pretrained/checkpoints/checkpoint_5000.ckpt --output gen_passwords.txt --batch-size 1024 --num-samples 1000000")
   print("passgan generator --input-dir pretrained --checkpoint pretrained/checkpoints/checkpoint_5000.ckpt --output gen_passwords.txt --batch-size 1024 --num-samples 1000000")
   print("passgan train --output-dir pretrained --training-data data/train.txt")
   print("passgan.py evaluate --input-dir pretrained --checkpoint pretrained/checkpoints/checkpoint_5000.ckpt --input-file passwords.txt --output scores.txt --batch-size 64")

def main(args=None):

      
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action='store_true', help="show this help message and exit")
    subparsers = parser.add_subparsers(dest="cmd")
    
    # 样本生成命令的参数
    subp_sample = subparsers.add_parser("sample", help='use the pretrained model to generate passwords', add_help=False)
    subp_generator = subparsers.add_parser("generator", help='use the pretrained model to generate passwords', add_help=False)
    subp_train = subparsers.add_parser("train", help='train a model on a large dataset (can take several hours on a GTX 1080)', add_help=False)
    # Evaluation command arguments
    subp_evaluate = subparsers.add_parser("evaluate", help='evaluate password strength using the discriminator', add_help=False)
    
    subp_sample.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    subp_sample.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .ckpt file.')

    subp_sample.add_argument('--output', '-o',
                        default='samples.txt',
                        help='File path to save generated samples to (default: samples.txt)')

    subp_sample.add_argument('--num-samples', '-n',
                        type=int,
                        default=1000000,
                        dest='num_samples',
                        help='The number of password samples to generate (default: 1000000)')

    subp_sample.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    subp_sample.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')
    
    subp_sample.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')


    subp_generator.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    subp_generator.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .ckpt file.')

    subp_generator.add_argument('--output', '-o',
                        default='samples.txt',
                        help='File path to save generated samples to (default: samples.txt)')

    subp_generator.add_argument('--num-samples', '-n',
                        type=int,
                        default=1000000,
                        dest='num_samples',
                        help='The number of password samples to generate (default: 1000000)')

    subp_generator.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    subp_generator.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')
    
    subp_generator.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')

    subp_generator.add_argument('--input-file', '-f',
                        required=True,
                        dest='input_file',
                        help='File containing passwords to generator (one per line).')

    # 训练命令的参数
    subp_train.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    subp_train.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    subp_train.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    subp_train.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    subp_train.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    subp_train.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    subp_train.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    
    subp_train.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    subp_train.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')
    # Evaluation command arguments
    subp_evaluate.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    subp_evaluate.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for evaluation. Expects a .ckpt file.')

    subp_evaluate.add_argument('--input-file', '-f',
                        required=True,
                        dest='input_file',
                        help='File containing passwords to evaluate (one per line).')

    subp_evaluate.add_argument('--output', '-o',
                        default='scores.txt',
                        help='File path to save evaluation results (default: scores.txt)')

    subp_evaluate.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')

    subp_evaluate.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')

    subp_evaluate.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the discriminator. Use the same value that you did for training (default: 128)')
    parsed_args = parser.parse_args(args)

    if parsed_args.cmd == "sample" :
        
        if not Path(parsed_args.input_dir).is_dir():
            parser.error(f'"{parsed_args.input_dir}" folder doesn\'t exist')

        if not Path(f'{parsed_args.checkpoint}.meta').is_file():
            parser.error(f'"{parsed_args.checkpoint}.meta" file doesn\'t exist')

        input_dir = Path(parsed_args.input_dir)

        if not (input_dir / 'charmap.pickle').is_file():
            parser.error(f'charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        if not (input_dir / 'inv_charmap.pickle').is_file():
            parser.error(f'inv_charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        sample_run(parsed_args)

    elif parsed_args.cmd == "generator":
        if not Path(parsed_args.input_dir).is_dir():
            parser.error(f'"{parsed_args.input_dir}" folder doesn\'t exist')

        if not Path(f'{parsed_args.checkpoint}.meta').is_file():
            parser.error(f'"{parsed_args.checkpoint}.meta" file doesn\'t exist')

        input_dir = Path(parsed_args.input_dir)

        if not (input_dir / 'charmap.pickle').is_file():
            parser.error(f'charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        if not (input_dir / 'inv_charmap.pickle').is_file():
            parser.error(f'inv_charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        generator_run(parsed_args)

    elif parsed_args.cmd == "train":
        train_run(parsed_args)
    elif parsed_args.cmd == "evaluate":
        if not Path(parsed_args.input_dir).is_dir():
            parser.error(f'"{parsed_args.input_dir}" folder doesn\'t exist')

        if not Path(f'{parsed_args.checkpoint}.meta').is_file():
            parser.error(f'"{parsed_args.checkpoint}.meta" file doesn\'t exist')

        input_dir = Path(parsed_args.input_dir)

        if not (input_dir / 'charmap.pickle').is_file():
            parser.error(f'charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        if not (input_dir / 'inv_charmap.pickle').is_file():
            parser.error(f'inv_charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        evaluate_run(parsed_args)
    elif parsed_args.help or (not(parsed_args.cmd)):
        help()
        exit()

if __name__ == "__main__":
    print(f"\n##################PassGAN##################\n")  
    main()
