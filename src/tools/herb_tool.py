from langchain.tools import tool
import os
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@tool
def tcm_herb_recognition(image_path: str) -> str:
    """
    真实视觉识别中药材工具（彻底不依赖文件名）。
    已强制要求：第一句话必须是“识别结果为：XXX”，后面再补充其他信息。
    """
    # 清理路径（彻底解决空格、换行、Windows反斜杠问题）
    clean_path = str(Path(image_path.strip()).resolve())
    print(f"🔍 最终使用的图片路径: {clean_path}")

    try:
        # 使用 .env 中的视觉模型
        vision_llm = ChatOpenAI(
            model=os.getenv("VISION_MODEL"),
            openai_api_key=os.getenv("API_KEY"),
            openai_api_base=os.getenv("BASE_URL"),
            temperature=0.1,
            max_tokens=1200
        )

        # 读取图片并转为 base64
        with open(clean_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # 【关键加强】强制第一句话必须是“识别结果为：XXX”
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "你是一位极其专业的中医专家。请仔细观察图片中的中药材，准确识别其名称（必须给出中文名）。\n"
                        "**严格要求**：你的输出**第一句话必须直接是**：识别结果为：XXX（XXX替换为正确的药材中文名称，例如“识别结果为：蒲公英”），不要加任何其他前缀、标题或【】。\n"
                        "然后再按以下格式继续回答：\n"
                        "1. 性味归经\n"
                        "2. 主要功效与主治\n"
                        "3. 一个简单实用的食疗方示例（配伍、做法、适用人群）\n"
                        "最后加一句：如需针对具体症状进行辨证调理，请提供更多信息。"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )

        response = vision_llm.invoke([message])
        return f"{response.content}"   # 直接返回模型输出，第一句话已强制为“识别结果为：XXX”

    except Exception as e:
        return (
            f"❌ 识别出现小问题：{str(e)}\n\n"
            "建议：\n"
            "1. 确保图片清晰、正对镜头、背景干净\n"
            "2. 检查 .env 中 VISION_MODEL 是否为 Qwen/Qwen2.5-VL-7B-Instruct\n"
            "3. 重新上传一次图片\n"
            "我已准备好作为中医专家继续为您解答！"
        )
