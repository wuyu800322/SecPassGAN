#Before running the code, please ensure that the following compressed files are extracted:

#PassGAN/hashpwd/hashed_gen_passwords.txt.zip
#PassGAN/data/train.txt.zip
#After extraction, you should have:

unzip PassGAN/hashpwd/hashed_gen_passwords.txt.zip -d PassGAN/hashpwd/
unzip PassGAN/data/train.txt.zip -d PassGAN/data/

cd /Users/xiakejie/PassMon/PassMon/PassGAN

source ./passganenv/bin/activate

source ~/.bashrc

cd /Users/xiakejie/PassMon/PassMon/flask_app

 python ./app.py
