import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import time
import os
import multiprocessing
import warnings
from torch.multiprocessing import Pool, set_start_method
import torch.multiprocessing as mp
import random
import string
import csv

# warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_cores = multiprocessing.cpu_count()
used_cores = max(1, int(available_cores * 0.8))
torch.set_num_threads(used_cores)

LOG_LEVEL = 2
if multiprocessing.current_process().name == "MainProcess":
    if LOG_LEVEL > 0:
        print(f"Using device: {DEVICE}")

try:
    set_start_method('spawn')
except RuntimeError:
    pass 

def init_worker(worker_id):
    import sys
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

class PasswordDataset(Dataset):
    def __init__(self, passwords, max_len):
        self.passwords = passwords
        self.max_len = max_len
        self.char_to_idx = {chr(i): i - 31 for i in range(32, 127)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<START>'] = len(self.char_to_idx)
        self.char_to_idx['<END>'] = len(self.char_to_idx)
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        password = self.passwords[idx]
        encoded = [self.char_to_idx['<START>']]
        for c in password:
            if c in self.char_to_idx:
                encoded.append(self.char_to_idx[c])
        encoded.append(self.char_to_idx['<END>'])
        encoded += [self.char_to_idx['<PAD>']] * (self.max_len - len(encoded))
        return torch.tensor(encoded[:self.max_len], dtype=torch.long)

class PasswordGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super(PasswordGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.fc(x)


def save_passwords_to_csv(passwords, output_file):
    # 将密码保存到CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Password"])  # 写入CSV标题
        for password in passwords:
            writer.writerow([password])  # 每个密码作为一行写入

    print(f"Passwords saved to {output_file}")

# Train the model
def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_epochs, patience=5):
    model.train()
    best_val_loss = float('inf')
    no_improve = 0
    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(num_epochs):
        start_time = time.time()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = loss_fn(outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        if LOG_LEVEL > 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        current_lr = optimizer.param_groups[0]['lr']
        if LOG_LEVEL > 0:
            print(f"Current learning rate: {current_lr}")

        scheduler.step(avg_val_loss)
        
        if current_lr != optimizer.param_groups[0]['lr']:
            if LOG_LEVEL > 0:
                print(f"Learning rate changed to: {optimizer.param_groups[0]['lr']}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                if LOG_LEVEL > 0:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        model.train()
    
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    return model

# Generate passwords in a batch
def generate_batch(model, char_to_idx, idx_to_char, batch_size, gpu_id, min_len=10, max_len=26, temp=1.0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    passwords = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    with torch.no_grad():
        current_chars = torch.tensor([[char_to_idx['<START>']] for _ in range(batch_size)], dtype=torch.long).to(device)

        for _ in range(max_len):
            outputs = model(current_chars)
            outputs = outputs[:, -1, :] / temp
            probs = torch.softmax(outputs, dim=-1)
            next_chars = torch.multinomial(probs, 1).squeeze(1)

            for i, next_char in enumerate(next_chars):
                if next_char == char_to_idx['<END>'] and len(passwords[i]) >= min_len:
                    finished[i] = True
                elif next_char != char_to_idx['<PAD>'] and not finished[i]:
                    passwords[i].append(idx_to_char[next_char.item()])

            if all(finished):
                break

            current_chars = torch.cat([current_chars, next_chars.unsqueeze(1)], dim=1)

    return [''.join(pwd) for pwd in passwords if len(pwd) >= min_len]

def ensure_min_length(passwords, min_len=10):
    # 用于补齐的字符集：字母和数字
    alphanumeric_chars = string.ascii_letters + string.digits
    
    # 对于每个密码，检查长度并补齐不足的部分
    padded_passwords = []
    for pwd in passwords:
        if len(pwd) < min_len:
            # 计算需要补齐的长度
            padding_needed = min_len - len(pwd)
            # 随机选择字符补齐
            pwd += ''.join(random.choices(alphanumeric_chars, k=padding_needed))
        padded_passwords.append(pwd)

    return padded_passwords

# Generate passwords with multiprocessing
def generate_passwords(model, char_to_idx, idx_to_char, num_passwords, batch_size, num_workers, temp):
    total_generated = 0
    passwords = []
    num_gpus = torch.cuda.device_count()

    model = model.cpu()

    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(total=num_passwords, desc="Generating passwords")
        
        while total_generated < num_passwords:
            batch_results = [pool.apply_async(generate_batch, (model, char_to_idx, idx_to_char, batch_size, i % num_gpus, temp)) for i in range(num_workers)]
            for result in batch_results:
                batch_passwords = result.get()
                passwords.extend(batch_passwords)
                new_passwords = len(batch_passwords)
                total_generated += new_passwords
                pbar.update(new_passwords)
        
        pbar.close()

    return passwords[:num_passwords]

def generate_input_batch(model, char_to_idx, idx_to_char, batch_size, gpu_id, input_password="", min_len=10, max_len=12, temp=1.0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    passwords = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    # Encode the input password if provided
    if input_password:
        encoded_input = [char_to_idx['<START>']] + [char_to_idx[c] for c in input_password if c in char_to_idx]
    else:
        encoded_input = [char_to_idx['<START>']]

    # Ensure input length doesn't exceed max_len
    input_len = len(encoded_input) - 1
    if input_len >= max_len:
        raise ValueError(f"Input password length exceeds maximum length: {max_len}")

    remaining_len = max_len - input_len  # Adjust the generation length based on the input password
    if LOG_LEVEL > 1:
        #print(f"max_len={max_len}")
        print(f"input_len={input_len}")
        #print(f"remaining_len={remaining_len}")
        #print(f"encoded_input={encoded_input}")
    with torch.no_grad():
        # Initialize current_chars with the encoded input
        current_chars = torch.tensor([encoded_input for _ in range(batch_size)], dtype=torch.long).to(device)

        for _ in range(remaining_len):  # Generate remaining characters to complete the password
            outputs = model(current_chars)
            outputs = outputs[:, -1, :] / temp
            probs = torch.softmax(outputs, dim=-1)
            next_chars = torch.multinomial(probs, 1).squeeze(1)

            for i, next_char in enumerate(next_chars):
                if next_char == char_to_idx['<END>'] and len(passwords[i]) >= min_len:
                    finished[i] = True
                elif next_char != char_to_idx['<PAD>'] and not finished[i]:
                    if next_char != char_to_idx['<END>']:
                        passwords[i].append(idx_to_char[next_char.item()])

            if all(finished):
                break
            current_chars = torch.cat([current_chars, next_chars.unsqueeze(1)], dim=1)

    generate_ps = list(set([''.join(pwd) for pwd in passwords if len(pwd) >= min_len]))
    random.shuffle(generate_ps)
    generate_ps_len = len(generate_ps)
    if LOG_LEVEL > 3:
        print(f"generate_ps len = {generate_ps_len}")
        print(f"generate_ps = {generate_ps}")
    return [input_password + pwd for pwd in generate_ps if len(pwd) >= min_len]
    #return [''.join(pwd) for pwd in passwords if len(pwd) >= min_len]
    
def generate_input_passwords(model, char_to_idx, idx_to_char, num_passwords, batch_size, num_workers, temp, input_password=""):
    total_generated = 0
    passwords = []
    num_gpus = torch.cuda.device_count()

    # 使用CPU进行模型推理
    model = model.cpu()

    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(total=num_passwords, desc="Generating input passwords")

        while total_generated < num_passwords:
            # 如果没有GPU，设置设备为0（表示CPU）
            if num_gpus > 0:
                batch_results = [pool.apply_async(generate_input_batch, (model, char_to_idx, idx_to_char, batch_size, i % num_gpus, input_password, temp)) for i in range(num_workers)]
            else:
                batch_results = [pool.apply_async(generate_input_batch, (model, char_to_idx, idx_to_char, batch_size, 0, input_password, temp)) for i in range(num_workers)]

            for result in batch_results:
                batch_passwords = result.get()
                passwords.extend(batch_passwords)
                new_passwords = len(batch_passwords)
                total_generated += new_passwords
                pbar.update(new_passwords)

        pbar.close()

    return passwords[:num_passwords]

# Add the Password Strength Check functions
import zxcvbn
def check_password_strength(password):
    #Use python zxcvbn module to evaluate Password Strength
    result = zxcvbn.zxcvbn(password)
    return result['score']

def classify_and_save_passwords(passwords):
    # classifications files initialize
    classifications = {
        "very_weak": [],
        "weak": [],
        "moderate": [],
        "strong": [],
        "very_strong": []
    }

    # evaluate every Password Strength and classify them
    for password in passwords:
        score = check_password_strength(password)
        if score == 0:
            classifications["very_weak"].append(password)
        elif score == 1:
            classifications["weak"].append(password)
        elif score == 2:
            classifications["moderate"].append(password)
        elif score == 3:
            classifications["strong"].append(password)
        elif score == 4:
            classifications["very_strong"].append(password)

    # save the classification result to different name files.
    # for category, passwords in classifications.items():
    #     with open(f"{category}_passwords.txt", 'w') as file:
    #         for password in passwords:
    #             file.write(f"{password}\n")

    return classifications

# Main entry point
def main():
    print("\n##################passmon##################\n")
    parser = argparse.ArgumentParser(description="Password Generator")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Mode: train or generate")
    parser.add_argument('--input', type=str, help="Input file for training")
    parser.add_argument('--output', type=str, help="Output file for generated passwords")
    parser.add_argument('--model', type=str, default='model.pth', help="Model file path")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=256, help="Batch size")
    parser.add_argument('--num_pass', type=int, default=100, help="Number of passwords to generate")
    parser.add_argument('--temp', type=float, default=1.0, help="Temperature for generation")
    parser.add_argument('--workers', type=int, default=4, help="Number of worker processes")
    parser.add_argument('--strength_check', type=int, default=1, help="Check the generated passwords file and output different strength ps files")
    parser.add_argument('--check_file', type=str, help="Input password file for strength evaluate")
    parser.add_argument('--input_password', type=str, help="Input password for seed")
    parser.add_argument('--log_level', type=int, default=1, help="Debug info print level, more details info with higher level")
    args = parser.parse_args()
    
    LOG_LEVEL = 2
    if args.mode == 'train':
        if not args.input:
            raise ValueError("Input file required for training")
        
        if os.path.exists(args.model):
            if LOG_LEVEL > 0:
                print(f"Loading model: {args.model}")
            checkpoint = torch.load(args.model, map_location=DEVICE)
            char_to_idx = checkpoint['char_to_idx']
            idx_to_char = checkpoint['idx_to_char']
            model = PasswordGenerator(len(char_to_idx), embed_size=256, hidden_size=512, num_layers=3)
            model.load_state_dict(checkpoint['model_state_dict'])
            if LOG_LEVEL > 0:
                print("Model loaded. Continuing training.")
        else:
            if LOG_LEVEL > 0:
                print("No model found. Initializing a new one.")
            char_to_idx = {chr(i): i - 31 for i in range(32, 127)}
            char_to_idx['<PAD>'] = 0
            char_to_idx['<START>'] = len(char_to_idx)
            char_to_idx['<END>'] = len(char_to_idx)
            idx_to_char = {i: char for char, i in char_to_idx.items()}
            model = PasswordGenerator(len(char_to_idx), embed_size=256, hidden_size=512, num_layers=3)

        model.to(DEVICE)

        with open(args.input, 'r', encoding='latin-1') as f:
            passwords = [line.strip() for line in f if 8 <= len(line.strip()) <= 26 and all(32 <= ord(c) < 127 for c in line.strip())]

        random.shuffle(passwords)
        max_len = 28
        train_passwords = passwords[:int(0.9 * len(passwords))]
        val_passwords = passwords[int(0.9 * len(passwords)):]

        train_dataset = PasswordDataset(train_passwords, max_len)
        val_dataset = PasswordDataset(val_passwords, max_len)

        train_dataset.char_to_idx = char_to_idx
        train_dataset.idx_to_char = idx_to_char
        train_dataset.vocab_size = len(char_to_idx)
        val_dataset.char_to_idx = char_to_idx
        val_dataset.idx_to_char = idx_to_char
        val_dataset.vocab_size = len(char_to_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True
        )

        if LOG_LEVEL > 0:
            print("Starting training...")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        model = train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, args.epochs)

        torch.save({
            'model_state_dict': model.state_dict(),
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char
        }, args.model)
        if LOG_LEVEL > 0:
            print(f"Model saved: {args.model}")

    elif args.mode == 'generate':
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        if not args.output:
            raise ValueError("Output file required for generation")

        if LOG_LEVEL > 0:
            print(f"Loading model: {args.model}")
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = checkpoint['idx_to_char']
        model = PasswordGenerator(len(char_to_idx), embed_size=256, hidden_size=512, num_layers=3)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        regenerated_flag = 0
        first_run = True
        times = 0
        temp = args.temp
        # with open(f"times.txt", 'w') as file_times:
        #     file_times.write(f"{times}\n")
        while (first_run or (regenerated_flag > 0)):
            if first_run:
                if LOG_LEVEL > 0:
                    print(f"Generating passwords...")
            else:
                if LOG_LEVEL > 0:
                    print(f"Regenerating passwords... on the {times} time")
            first_run = False
            regenerated_flag = 0
            if temp > 8.0:
                temp = temp - 0.2
            if LOG_LEVEL > 0:
                print(f"Temp : {temp:.2f} on the {times} time")
            if not args.input_password:
                passwords = generate_passwords(model, char_to_idx, idx_to_char, args.num_pass, args.batch, args.workers, temp)
                passwords = ensure_min_length(passwords)
            else:
                passwords = generate_input_passwords(model, char_to_idx, idx_to_char, args.num_pass, args.batch, args.workers, temp, args.input_password)
                passwords = ensure_min_length(passwords)

            # with open(args.output, 'w', encoding='utf-8') as f:
            #     for password in passwords:
            #         f.write(f"{password}\n")
            shuffled_passwords=[]
            with open(args.output, 'w', encoding='utf-8') as f:
                for password in passwords:
                    shuffled_password = ''.join(random.sample(password, len(password)))
                    shuffled_passwords.append(shuffled_password)
                    f.write(f"{shuffled_password}\n")
            
            csv_output_file = args.output.replace('.txt', '.csv')
            print('##############Generated Passwords##############')
            print(shuffled_passwords)
            print('###############################################')
            save_passwords_to_csv(passwords, csv_output_file)
            
            if LOG_LEVEL > 0:
                print(f"Passwords saved to: {args.output}")
            if args.input_password:
                if LOG_LEVEL > 2:
                    print(f"Generate new Passwords is: {passwords}")

            if args.strength_check != 0:
                if LOG_LEVEL > 1:
                    print(f"Loading Passwords file: {args.output}")
                # read passwords file
                with open(args.output, 'r') as file:
                    passwords = [line.strip() for line in file]

                if LOG_LEVEL > 1:
                    print(f"Begin to check Passwords file {args.check_file} strength!")
                # classify the passwords according the different strength and save
                classifications = classify_and_save_passwords(passwords)

                # output password strength classification results
                if LOG_LEVEL > 1:
                    print("The summary of password strength classification:")
                for category, passwords in classifications.items():
                    if LOG_LEVEL > 1:
                        print(f"{category.capitalize()}: {len(passwords)}")
                    if len(passwords) > 0:
                        # Check if the number of Weak passwords is greater than 0
                        if LOG_LEVEL > 2:
                            print(f"Passwords Type: {category.capitalize()}")
                        if ( category.capitalize() == "Weak"):
                            regenerated_flag = regenerated_flag + 1
                            temp = temp + 0.1
                            if LOG_LEVEL > 0:
                                print(f"Warning: There are {len(passwords)} Weak passwords in the list. Regenerate the passwords.")
                        # Check if the number of Very_weak passwords is greater than 0
                        if ( category.capitalize() == "Very_weak"):
                            regenerated_flag = regenerated_flag + 1
                            temp = temp + 0.2
                            if LOG_LEVEL > 0:
                                print(f"Warning: There are len(passwords) Very_weak passwords in the list. Regenerate the passwords.")
            if regenerated_flag > 0:
                times = times + 1
                # with open(f"times.txt", 'w') as file_times:
                #     file_times.write(f"{times}\n")

    elif args.strength_check != 0:
        if not args.check_file:
            raise ValueError("Input password file required for strength check!!!")
        if not os.path.exists(args.check_file):
            raise FileNotFoundError(f"Password file not found: {args.check_file}!!!")

        if LOG_LEVEL > 0:
            print(f"Loading Passwords file: {args.check_file}")
        # read passwords file
        with open(args.check_file, 'r') as file:
            passwords = [line.strip() for line in file]

        if LOG_LEVEL > 0:
            print(f"Begin to check Passwords file {args.check_file} strength!")
        # classify the passwords according the different strength and save
        classifications = classify_and_save_passwords(passwords)

        # output password strength classification results
        if LOG_LEVEL > 0:
            print("The summary of password strength classification:")
        for category, passwords in classifications.items():
            if LOG_LEVEL > 0:
                print(f"{category.capitalize()}: {len(passwords)}")

if __name__ == "__main__":
    main()
    print("Process completed successfully.")
