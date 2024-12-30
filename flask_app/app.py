from flask import Flask, g, request, jsonify, render_template
import subprocess
import zxcvbn
import os
import re
from collections import Counter
import math
import logging
import random
from tqdm import tqdm

app = Flask(__name__)
# 配置 Flask 的日志系统
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
app.logger.setLevel(logging.DEBUG)

# 设置全局字典
@app.before_request
def setup_global_dict():
    g.global_dict = {}
    g.old_pwds = []
    g.suggested_pwd = ''
    g.suggested_pwd_score = ''
    g.des_pwd_string = ''

def set_global_value(key,value):
    g.global_dict[key] = value
    return "Global value set."

def get_global_value(key):
    value = g.global_dict.get(key, '')
    return value

def get_best_password():
    """筛选与旧密码相似度最低且强度最高的密码，并返回最优密码"""
    if not g.global_dict:
        print("Global dictionary is empty.")
        return None

    # 确保 g.old_pwds 是数组
    if not isinstance(g.old_pwds, list) or len(g.old_pwds) == 0:
        print("Old passwords must be a non-empty list.")
        return None

    # 最优密码及评分
    best_password = None
    best_score = -1
    min_similarity = float('inf')  # 相似度最低值初始化为正无穷大

    # 遍历 global_dict 中的所有密码
    for idx, pwd in enumerate(g.global_dict.keys()):
        # 防止索引越界
        if idx < len(g.old_pwds):
            old_pwd = g.old_pwds[idx]  # 获取对应旧密码

            # 计算与旧密码的相似度
            similarity = cosine_similarity(pwd, old_pwd)
            print(f"Similarity between {pwd} and {old_pwd}: {similarity}")

            # 计算密码强度评分
            score = zxcvbn.zxcvbn(pwd)['score']

            # 筛选逻辑：
            # 1. 优先选择相似度最低的密码
            # 2. 若相似度相同，则选择评分最高的密码
            if similarity < min_similarity or (similarity == min_similarity and score > best_score):
                best_password = pwd
                best_score = score
                min_similarity = similarity  # 更新最小相似度
            elif similarity == min_similarity and score == best_score:
                # 强度和相似度都一致时随机选择
                if random.choice([True, False]):
                    best_password = pwd

    # 输出最终筛选结果
        
    g.suggested_pwd = best_password + f"({100*min_similarity:.0f}%)"
    g.suggested_pwd_score = ['very_weak', 'weak', 'fair', 'strong', 'very_strong'][best_score]
                                     
    print(f"Best password: {best_password}, Score: {best_score}, Similarity: {min_similarity}")
    return ""  # 返回最优密码

def cosine_similarity(password1, password2):
    
    vec1, vec2 = Counter(password1), Counter(password2)
    common_chars = set(vec1.keys()).intersection(set(vec2.keys()))
    
    numerator = sum(vec1[char] * vec2[char] for char in common_chars)
    sum1 = sum(count**2 for count in vec1.values())
    sum2 = sum(count**2 for count in vec2.values())
    
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    result = numerator / denominator if denominator else 0.0
    print('cosine_similarity:'+password1+'||'+password2+'='+str(result))
                            
    return result

@app.route("/")
def home():
    return render_template("base.html")

def run_passgan_evaluation(file_path):
    """
    使用 PassGAN 对 './UserPassword.txt' 中的密码进行评估，并将结果输出到 './PassganScore.txt'。
    如果评估成功，返回 True，否则返回 False 并打印错误信息。
    """
    # 检查输入文件是否存在
    if os.path.exists(file_path):
        # 构建 PassGAN 评估命令
        command = [
            'python', '../PassGAN/passgan.py', 'evaluate',
            '--input-dir', '../PassGAN/pretrained',
            '--checkpoint', '../PassGAN/pretrained/checkpoints/checkpoint_5000.ckpt',
            '--input-file', file_path,
            '--output',  file_path.replace('UserPassword', 'PassganScore'),
            '--batch-size', '64',
            '--layer-dim', '128'
        ]

        try:
            # 运行 PassGAN 评估命令
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print("PassGAN evaluation error:", e)
            return False
    else:
        print("Input file './UserPassword.txt' does not exist.")
        return False

def generate_passgan_score_summary(file_path='./PassganScore.txt',ignore_check=False):
    # 初始化 summary 字典
    summary = {
        'passgan_score_counts': {
            'Fake': 0,
            'Real': 0,  
        },
        'passgan_score_passwords': {
            'Fake': [],
            'Real': [],
        }
    }

    # 颜色映射
    color_map = {
        'Fake': '#ff4d4d',   # 红色
        'Real': '#128EE9',   # 黄色
    }

    # 分数区间标签映射
    score_ranges = {
        'Fake': '0.0~1.5',
        'Real': '1.5~3.0',
    }

    # 检查文件是否存在
    score_psws = []
    min_similarity = float('inf')
    min_score = float('inf')
   
    if os.path.exists(file_path):
        with open(file_path, 'r') as score_file:
            for line in score_file:
                # 使用正则表达式捕获密码和分数
         
                match = re.search(r"^(\S+)\s+\[([0-9.]+)\]", line)
                if match:
                    password = match.group(1)
                    score_psws.append(password)
                    
                    score = zxcvbn.zxcvbn(password)['score']
              
                    
                    if not ignore_check and score != 4:
                        continue

                    passgan_score = float(match.group(2))  # PassGAN 的分数
                    
                    # 将分数分类并计数，同时将密码和分数添加到对应分数的列表中
                    if 0.0 <= passgan_score < 1.5:
                        summary['passgan_score_counts']['Fake'] += 1
                        summary['passgan_score_passwords']['Fake'].append((password, passgan_score))
                        set_global_value(password,passgan_score)
 
                        
                    elif 1.5 <= passgan_score < 3.0:
                        summary['passgan_score_counts']['Real'] += 1
                        summary['passgan_score_passwords']['Real'].append((password, passgan_score))

    # 生成 HTML 输出
    g.des_pwd_string = "<h3>PassGAN Discriminator Results:</h3>"
    frame_summary = "<h3>PassGAN Discriminator && Cosine Similarity Results:</h3>"
    for score_key, passwords in summary['passgan_score_passwords'].items():
        color = color_map[score_key]
        count = summary['passgan_score_counts'][score_key]
        frame_summary += f"<h3 style='color: {color};'>{score_key} ({count})</h3>"
        if passwords:
                frame_summary += "<p>"
                for idx, (pwd, passgan_score) in enumerate(passwords):
 
                    # 添加密码及对应的分数
                    g.des_pwd_string += (
                        f"<span style='color: green;'>{pwd}</span>, "
                    )
                    print(pwd)
                    if pwd in score_psws:
                        if idx < len(g.old_pwds) :  # 确保索引不越界
                            old_pwd = g.old_pwds[idx]
                            similarity = cosine_similarity(pwd, old_pwd)
                            if similarity <= min_similarity and score_key == 'Fake':
                                 min_similarity = similarity
                                 print(pwd+"#######111######"+str(similarity))
                                #  g.suggested_pwd = pwd  + f"({100*similarity:.0f}%)"
                                 
                                 if passgan_score<=min_score:
                                     min_score = passgan_score
                                     print(score_key)
                                     
                                     sug_score = zxcvbn.zxcvbn(pwd)['score']
                                    #  g.suggested_pwd_score = ['very_weak', 'weak', 'fair', 'strong', 'very_strong'][sug_score]
                                     
                                
                            if score_key == 'Fake':
                                frame_summary += (
                                    f"<span style='color: {color};'>{pwd}"
                                    f"({100*similarity:.0f}%)</span>, "
                                    )
                            else:
                                frame_summary += (
                                    f"<span style='color: {color};'>{pwd}</span>," 
                                    )        

                        else:
                            # g.old_pwds 中索引越界时的处理
                            frame_summary += (
                                f"<span style='color: {color};'>{pwd} "
                                f"(No matching old password)({passgan_score:.4f})</span>, "
                            )
                    else:
                        frame_summary += (
                            f"<span style='color: {color};'>{pwd}"
                            f"(Not found in score_psws)({passgan_score:.4f})</span>, "
                        )

                # 去掉最后一个逗号并关闭段落
                frame_summary = frame_summary.rstrip(", ") + "</p>"

    return frame_summary

def extract_and_save_strong_passwords(user_file_path, output_file_path='strong_passwords.txt'):
    """
    提取强密码并保存到指定文件中。
    
    :param user_file_path: 包含用户密码的文件路径。
    :param output_file_path: 用于保存强密码的文件路径。
    """
    strong_passwords = []  # 用于存储强密码的数组

    # 检查输入文件是否存在
    if os.path.exists(user_file_path):
        with open(user_file_path, 'r') as f:
            passwords = f.read().splitlines()
        
        # 对每个密码计算强度并分类
        for password in passwords:
            try:
                score = zxcvbn.zxcvbn(password)['score']
                strength_key = ['very_weak', 'weak', 'fair', 'strong', 'very_strong'][score]
                
                # 如果强度包含 "very_strong"，将密码添加到 strong_passwords 数组中
                if "very_strong" in strength_key:
                    strong_passwords.append(password)
            except Exception as e:
                print(f"Error processing password '{password}': {e}")

  
    # 保存强密码到文件
    if strong_passwords:
        with open(output_file_path, 'w') as f:
            for password in strong_passwords:
                f.write(password + '\n')
    else:
        print("No strong passwords found.")
    return len(strong_passwords)

# def pwd_summary(file_path,name):
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             passwords = f.read().splitlines()
#             frame_summary = "<h3>"+name+"</h3>"
#             frame_summary += "<p>total_passwords:"+str(len(passwords))+"</p>"
#             frame_summary += "<p>max_length:"+str(max(len(p) for p in passwords))+"</p>"
#             frame_summary += "<p>min_length:"+str(min(len(p) for p in passwords))+"</p>"
#             frame_summary += "<p>avg_length:" + "{:.1f}".format(sum(len(p) for p in passwords) / len(passwords)) + "</p>"
#     return frame_summary

def pwd_assessment(file_path,name,record):
    """根据文件中的密码内容，计算并返回按强度分类的HTML格式内容。"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            passwords = f.read().splitlines()
            frame_content = "<h3>"+name+"</h3>"
        
        strong_passwords = []
        # 初始化按强度分组的字典
        password_groups = {
            'very_weak': [],
            'weak': [],
            'fair': [],
            'strong': [],
            'very_strong': []
        }

        # 对每个密码计算强度并分类
        for password in passwords:
            score = zxcvbn.zxcvbn(password)['score']
            strength_key = ['very_weak', 'weak', 'fair', 'strong', 'very_strong'][score]
            password_groups[strength_key].append(password)
            if "very_strong" in strength_key:
                strong_passwords.append(password) 
        if record:
            g.old_pwds = strong_passwords
        # 颜色映射
        color_map = {
            'very_weak': '#ff4d4d',  # 红色
            'weak': '#ff9933',       # 橙色
            'fair': '#ffd633',       # 黄色
            'strong': '#80cc33',     # 绿色
            'very_strong': '#339966' # 深绿色
        }

        # 格式化输出并设置颜色
        frame_content = "<h3>" + name + "</h3>"
        for strength, pwd_list in password_groups.items():
            color = color_map[strength]
            frame_content += f"<p style='color: {color};'>{strength.capitalize()} ({len(pwd_list)})</p>"

            if pwd_list:
                # 添加密码和相似度信息
                frame_content += "<p>" + ", ".join([
                    f"<span style='color: {color};'>{pwd} </span>"
                    for pwd in pwd_list
                ]) + "</p>"
            else:
                frame_content += "<p style='color: #ccc;'>None</p>"
        return frame_content
    else:
        return "<p style='color: #ccc;'>Password file does not exist or cannot be read.</p>"


def run_password_generation(command, total_steps=None):
    """
    运行密码生成命令并实时打印生成的日志，同时显示进度条。

    Args:
        command (list): 要运行的命令列表，例如 ['python', 'script.py', '--arg', 'value']。
        total_steps (int, optional): 预计总步数，用于显示进度条。如果不传递，则不显示进度条。
        
    Returns:
        None
    """
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # 行缓冲模式
        )
        
        # 如果提供了总步数，则初始化进度条
        progress_bar = None
        if total_steps is not None:
            progress_bar = tqdm(total=total_steps, desc="Progress", unit="step")

        # 读取并处理输出
        with process.stdout as stdout, process.stderr as stderr:
            for line in iter(stdout.readline, ""):
                print(line.strip())  # 打印日志
                # 检查是否包含进度信息，并更新进度条
                if progress_bar and "progress:" in line.lower():
                    # 假设日志中包含类似 "progress: 5/100" 的信息
                    progress_match = re.search(r"progress:\s*(\d+)/(\d+)", line, re.IGNORECASE)
                    if progress_match:
                        current_step = int(progress_match.group(1))
                        progress_bar.n = current_step
                        progress_bar.refresh()
            
            for line in iter(stderr.readline, ""):
                print(f"STDERR: {line.strip()}")

        return_code = process.wait()
        if progress_bar:
            progress_bar.close()

        if return_code != 0:
            print(f"Command failed with return code: {return_code}")
    except Exception as e:
        print(f"Unexpected error while running command: {e}")

@app.route("/generate-passwords", methods=["POST"])
def generate_passwords():
    logging.info("\n\n################################Strong Password Generation System################################\n\n")
    
    # 获取表单数据
    num_pass = request.form.get("num_pass", "16")
    input_sentence = request.form.get("input_password", "")
    logging.info(f"Received input: num_pass={num_pass}, input_password={input_sentence}")
    
    model_path = './Model/model_rockyou-train_strong_E50_B512_W32_20241009_155026.pth'
    g.old_pwds = []
    g.suggested_pwd_score = ''
    
    # 提取密码的前缀
    if input_sentence:
        words = input_sentence.split()
        if len(words) == 1:
            input_password = words[0][:6]  # 取单词的前6个字符
        else:
            input_password = ''.join([word[0] for word in words[:6]])
    else:
        input_password = ''
    
    logging.info(f"Generated input_password prefix: {input_password}")
    
    temp = "0.5"
    file_name1 = './LSTM_UserPassword.txt'
    
    # 定义第一个生成命令
    cmd1 = [
        'python', '../passmon.py',
        '--mode', 'generate',
        '--model', model_path,
        '--output', file_name1,
        '--num_pass', num_pass,
        '--temp', temp,
        '--workers', '1',
        '--input_password', input_password,
        "--log_level", "2"
    ]
    
    logging.info(f"Executing command: {cmd1}")
    run_password_generation(cmd1,total_steps=200)
    logging.info(f"Command completed. Output file: {file_name1}")
    
    psws_length = extract_and_save_strong_passwords(file_name1, 'UserPassword_GAN0.txt')
    logging.info(f"Extracted strong passwords: {psws_length} strong passwords found.")
    
    cmdM = [
        "python", "../PassGAN/passgan.py", "generator",
        "--input-dir", "../PassGAN/pretrained",
        "--checkpoint", "../PassGAN/pretrained/checkpoints/checkpoint_5000.ckpt",
        "--input-file", 'UserPassword_GAN0.txt',
        "--output", file_name1,
        "--batch-size", str(psws_length),
        "--num-samples", str(psws_length)
    ]
    
    logging.info(f"Executing second command: {cmdM}")
    frame1_content = pwd_assessment(file_name1, 'PassMon Generated Passwords:', True)
    logging.info("\n#########################")
    logging.info(frame1_content)
    logging.info("\n#########################\n")
    logging.info("\n##########abc###############\n"+file_name1)

    run_passgan_evaluation(file_name1)
    score1_file = file_name1.replace('UserPassword', 'PassganScore')
    frame1_mon_score =  generate_passgan_score_summary(score1_file, False)
    logging.info("\n##########abc###############\n"+frame1_mon_score)

 
    if os.path.exists(file_name1):
        os.rename(file_name1, 'UserPassword_GAN1.txt')
        logging.info(f"Renamed output file to 'UserPassword_GAN1.txt'")
    

    run_password_generation(cmdM,total_steps=200)
    
    frame1_passgan_pwd = pwd_assessment(file_name1, 'PassGan Generated Passwords:', False)
    logging.info("\n#########################")
    logging.info(frame1_passgan_pwd)
    logging.info("\n#########################\n")
    
 
    run_passgan_evaluation(file_name1)
    
    score1_file = file_name1.replace('UserPassword', 'PassganScore')
    frame1_score = generate_passgan_score_summary(score1_file, True)
   

    get_best_password()

    logging.info("\n#########################")
    logging.info(frame1_score)
    logging.info("\n#########################\n")
    frame1_suggested = g.suggested_pwd
    
    if os.path.exists(file_name1):
        os.remove(file_name1)  # 清理文件
  
    
    if os.path.exists(score1_file):
        os.remove(score1_file)  # 清理文件
     
    logging.info("\n\n##################################Suggested Password###################################\n\n")
    logging.info(f"Suggested passwords: {frame1_suggested}")
    logging.info("\n\n##################################Suggested Password###################################\n\n")
    
 

 
    # 返回生成的密码和分类信息给前端
    return jsonify({
        "frame1Content": frame1_content,
        "frame1MonScore": frame1_mon_score,
        "frame1PassGAN": frame1_passgan_pwd,
        "frame1Score": frame1_score,
        "frame1Suggested": frame1_suggested,
        "frame1SuggestedScore": g.suggested_pwd_score,
        "frame1Discriminator": g.des_pwd_string,
    })
if __name__ == "__main__":
    app.run(debug=True)
