# main.py

# 导入 callback 包
from callbacks import add_integration_callbacks, default_callbacks, get_default_callbacks

# 使用导入的符号
print("Default callbacks:", default_callbacks)
callbacks = get_default_callbacks()
print("Callbacks:", callbacks)

# 调用函数
add_integration_callbacks(callbacks)
