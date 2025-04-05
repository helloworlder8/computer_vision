import re

text = "This  is    a\nstring   with \t irregular \n whitespace."
text = re.sub(r"\s+", " ", text)
print(text)
# 输出: "This is a string with irregular whitespace."


import ftfy

text = "This is an invalid encoding: â€œHelloâ€"
fixed_text = ftfy.fix_text(text)
print(fixed_text)
# 输出: This is an invalid encoding: “Hello”

import html

text = "&amp;lt;div&amp;gt;Hello&amp;lt;/div&amp;gt;"
unescaped_text = html.unescape(html.unescape(text))
print(unescaped_text)
# 输出: <div>Hello</div>


text = "   Hello World!   "
cleaned_text = text.strip()
print(cleaned_text)
# 输出: "Hello World!"



import ftfy
import html

def clean_text(text):
    text = ftfy.fix_text(text)  # 修复编码错误
    text = html.unescape(html.unescape(text))  # 解码 HTML 实体（包括嵌套）
    return text.strip()  # 移除首尾空白

# 测试输入
raw_text = "   &amp;lt;p&amp;gt;Hello, â€œworld!â€&amp;lt;/p&amp;gt;   "
cleaned_text = clean_text(raw_text)
print(cleaned_text)
# 输出: <p>Hello, “world!”</p>
