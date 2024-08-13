import subprocess
import sys
import argparse
from datetime import datetime

# 检查是否初始化了 Git 仓库
def is_git_repository():
    try:
        subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'])
        return True
    except subprocess.CalledProcessError:
        return False

# 检查是否存在同名分支
def branch_exists(branch_name):
    try:
        subprocess.check_output(['git', 'show-ref', '--verify', '--quiet', f'refs/heads/{branch_name}'])
        return True
    except subprocess.CalledProcessError:
        return False

# 主函数
def main(args):


    # 获取远程仓库URL
    remote_repo_url = args.remote_repo_url

    # 如果没有传递提交信息参数，使用当前日期的 "年-月-日" 作为提交信息
    if args.push_message is None:
        pushmessage = datetime.now().strftime('%y-%m-%d')
    else:
        # 如果传递了提交信息参数，使用传递的参数作为提交信息
        pushmessage = args.push_message

    print(f"Push message: {pushmessage}")

    # 初始化 Git 仓库（如果未初始化）
    if not is_git_repository():
        print("Initializing a new Git repository...")
        subprocess.check_call(['git', 'init'])
        subprocess.check_call(['git', 'remote', 'add', 'origin', remote_repo_url])

    # 切换或创建分支
    if branch_exists(pushmessage):
        # 如果分支已经存在，切换到该分支
        subprocess.check_call(['git', 'checkout', pushmessage])
    else:
        # 如果分支不存在，创建并切换到该分支
        subprocess.check_call(['git', 'checkout', '-b', pushmessage])

    # 添加所有的更新
    subprocess.check_call(['git', 'add', '-A'])

    # 提交更改
    subprocess.check_call(['git', 'commit', '-m', pushmessage])

    # 推送到远程仓库并设置上游分支
    try:
        subprocess.check_call(['git', 'push', '-u', 'origin', pushmessage])
    except subprocess.CalledProcessError:
        print("Error pushing to remote repository. Make sure the remote repository URL is correct and you have the necessary permissions.")

# 使用 if __name__ == '__main__' 语句来调用 main 函数
if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Push to a remote Git repository with a custom branch name and commit message.")
    parser.add_argument('remote_repo_url', nargs='?', default="https://github.com/helloworlder8/computer_vision.git", help="The URL of the remote Git repository. Default is 'https://github.com/helloworlder8/computer_vision.git'")
    parser.add_argument('push_message', nargs='?', default=None, help="The commit message. If not provided, the current date will be used.")

    # 解析参数
    args = parser.parse_args()
    main(args)

# 命令行输入
# python push.py https://github.com/helloworlder8/ultralytics-8.2.60.git


# 远程删除
# git push origin --delete 24-07-31
# 本地删除
# git branch -d 24-07-31

# 手动推送
# git init
# git remote add origin https://github.com/helloworlder8/computer_vision.git
# git branch
# git add -A
# git commit -m "24-08-01"
# git push -u origin 24-08-13
# git config --global --unset http.proxy
# git config --global --unset https.proxy
#helloworlder8
#ghp_3evWUSgNgYio9UWL4H0C6yxqgrU33x4FUlww

# 调试文件
# launch.json文件
# {
#     // 使用 IntelliSense 了解相关属性。 
#     // 悬停以查看现有属性的描述。
#     // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python 调试程序: 当前文件",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "${file}",
#             "console": "integratedTerminal",
#             "args": [
#                 "https://github.com/helloworlder8/ultralytics-8.2.60.git",
#             ]
#         }
#     ]
# }