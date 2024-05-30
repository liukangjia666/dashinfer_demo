# coding: utf-8 
# @Time    :
# @Author  :
# @description :
from typing import Tuple

class Role:
    USER = 'user'
    SYSTEM = 'system'
    BOT = 'bot'
    ASSISTANT = 'assistant'
    ATTACHMENT = 'attachment'


default_system = 'You are a helpful assistant.'
def history_to_messages(history, system: str):
    """
    格式化传参中的history和系统提示词
    :param history: [[问题,回答],[问题,回答]]
    :param system:
    :return:
    """
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages


def messages_to_history(messages):
    """
    历史聊天保留
    :param messages:
    :return:
    """
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history


def message_to_prompt(messages) -> str:
    """
    提示词模板格式化
    :param messages:
    :return:
    """
    prompt = ""
    for item in messages:
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        prompt += f"\n{im_start}{item['role']}\n{item['content']}{im_end}"
    prompt += f"\n{im_start}assistant\n"
    return prompt

