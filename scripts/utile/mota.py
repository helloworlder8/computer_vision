
# from modelscope.hub.api import HubApi
# YOUR_ACCESS_TOKEN = 'e00292e3-3e13-4bde-9ff1-40d15c74f9d2'
# api = HubApi()
# api.login(YOUR_ACCESS_TOKEN)

# from modelscope.hub.constants import Licenses, ModelVisibility

# username = 'helloworlder8'
# model_name = 'ALSS-YOLO-Seg'

# api.create_model(
#     model_id=f"{username}/{model_name}",
#     visibility=ModelVisibility.PUBLIC,
#     license=Licenses.APACHE_V2,
#     chinese_name="",
# )

from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'e00292e3-3e13-4bde-9ff1-40d15c74f9d2'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

# 上传模型
api.push_model(
    model_id="helloworlder8/ALSS-YOLO-Seg", # 如果model_id对应的模型库不存在，将会自动创建
    model_dir="./" # 指定本地模型目录，目录中必须包含configuration.json文件
)