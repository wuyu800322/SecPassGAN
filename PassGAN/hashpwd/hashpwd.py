import hashlib

# 输入的密码文件路径（PassGAN 生成的密码文件）
input_password_file = 'gen_passwords.txt'

# 输出的哈希文件路径（Hashcat 需要的文件）
output_hash_file = 'hashed_gen_passwords.txt'

# 选择使用的哈希算法（以 MD5 为例）
def hash_password(password):
    # 使用 hashlib 生成 MD5 哈希
    return hashlib.md5(password.encode('utf-8')).hexdigest()

# 读取 PassGAN 生成的密码文件
with open(input_password_file, 'r') as infile:
    passwords = infile.readlines()

# 打开 hashfile.txt 以写入生成的哈希值
with open(output_hash_file, 'w') as outfile:
    for password in passwords:
        password = password.strip()  # 去除换行符
        hashed_password = hash_password(password)
        outfile.write(f"{hashed_password}\n")

print(f"哈希文件已生成：{output_hash_file}")
