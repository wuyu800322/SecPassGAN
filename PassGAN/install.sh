#!/bin/sh

# 确保 /usr/local/bin/passgan 是一个文件而不是目录
if [ -d "/usr/local/bin/passgan" ]; then
    sudo rm -rf "/usr/local/bin/passgan"
fi

# 创建安装目录
install -dm 755 "/usr/local/share/passgan"

# 安装文档和许可证
install -Dm 644 -t "/usr/local/share/doc/passgan/" README.md 2>/dev/null
install -Dm 644 LICENSE "/usr/local/share/licenses/passgan/LICENSE" 2>/dev/null

# 删除文档和许可证
rm -rf README.md LICENSE

# 复制文件到目标目录
cp -a * "/usr/local/share/passgan"

# 创建可执行脚本
cat > "/usr/local/bin/passgan" << EOF
#!/bin/sh

exec python /usr/local/share/passgan/passgan.py "\$@"
EOF

# 给可执行文件赋予权限
chmod a+x "/usr/local/bin/passgan"

# 打印安装成功消息
echo "PassGAN installed."
