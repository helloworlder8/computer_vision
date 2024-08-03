# python script/train.py;python train1.py ;python train2.py; python send_notice.py
# python train3.py;python train4.py;python train5.py;
# # 假设您的程序原执行命令为
python train1.py; 
# python train2.py; python train3.py

# # 那么可以在您的程序后跟上shutdown命令
# python train1.py; /usr/bin/shutdown      # 用;拼接意味着前边的指令不管执行成功与否，都会执行shutdown命令
# python train.py && /usr/bin/shutdown    # 用&&拼接表示前边的命令执行成功后才会执行shutdown。请根据自己的需要选择


# import os

# if __name__ == "__main__":
#     # xxxxxx
#     os.system("/usr/bin/shutdown")