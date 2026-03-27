from langchain.tools import tool
import os


@tool
def tcm_herb_recognition(image_path: str) -> str:
    """
    识别中药材图片的工具。输入图片路径，返回药材名称及其基本中医功效。
    支持识别：百合(baihe)、党参(dangshen)、枸杞(gouqi)、槐花(huaihua)、金银花(jinyinhua)。
    """
    # 架构师笔记：在实际生产中，这里会调用一个 CNN 模型或 GPT-4o-Vision API。
    # 为了演示方便，我们根据文件名或路径关键字来模拟识别逻辑。

    path_lower = image_path.lower()

    herb_data = {
        "baihe": "【百合】：养阴润肺，清心安神。主治阴虚久咳，失眠多梦。",
        "dangshen": "【党参】：补中益气，健脾益肺。用于脾肺虚弱，气短心悸。",
        "gouqi": "【枸杞】：滋补肝肾，益精明目。用于虚劳精亏，腰膝酸痛。",
        "huaihua": "【槐花】：凉血止血，清肝泻火。用于便血，痔血，肝火头痛。",
        "jinyinhua": "【金银花】：清热解毒，疏散风热。用于外感风热，疮痈肿毒。"
    }

    # 模拟匹配逻辑
    for key in herb_data:
        if key in path_lower:
            return f"识别成功！该药材是：{herb_data[key]}"

    return "抱歉，该药材不在我的识别库中（目前仅支持：百合、党参、枸杞、槐花、金银花）。"