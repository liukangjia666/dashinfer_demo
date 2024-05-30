# coding: utf-8 
# @Time    :
# @Author  :
# @description :
import os
import copy
import threading
import subprocess
import random
from typing import Optional

from llm_utils import history_to_messages, Role, message_to_prompt, messages_to_history

os.system("pip uninstall -y tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem")
os.environ["LANG"] = "C"
os.environ["LC_ALL"] = "C"

from dashinfer.helper import EngineHelper, ConfigManager

log_lock = threading.Lock()

config_file = "config_qwen_v15_1_8b.json"
config = ConfigManager.get_config_from_json(config_file)

def download_model(model_id, revision, source="modelscope"):
    """
    x下载原始模型
    :param model_id:
    :param revision:
    :param source:
    :return:
    """
    print(f"Downloading model {model_id} (revision: {revision}) from {source}")
    if source == "modelscope":
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id, revision=revision)
    elif source == "huggingface":
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=model_id)
    else:
        raise ValueError("Unknown source")

    print(f"Save model to path {model_dir}")

    return model_dir

cmd = f"pip show dashinfer | grep 'Location' | cut -d ' ' -f 2"
package_location = subprocess.run(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  shell=True,
                                  text=True)
package_location = package_location.stdout.strip()
os.environ["AS_DAEMON_PATH"] = package_location + "/dashinfer/allspark/bin"
os.environ["AS_NUMA_NUM"] = str(len(config["device_ids"]))
os.environ["AS_NUMA_OFFSET"] = str(config["device_ids"][0])

## download model from modelscope
original_model = {
    "source": "modelscope",
    "model_id": "qwen/Qwen1.5-1.8B-Chat",
    "revision": "master",
    "model_path": ""
}
original_model["model_path"] = download_model(original_model["model_id"],
                                              original_model["revision"],
                                              original_model["source"])

# GPU模型转换成CPU格式文件
engine_helper = EngineHelper(config)
engine_helper.verbose = True
engine_helper.init_tokenizer(original_model["model_path"])
## convert huggingface model to dashinfer model
## only one conversion is required
engine_helper.convert_model(original_model["model_path"])
engine_helper.init_engine()
engine_max_batch = engine_helper.engine_config["engine_max_batch"]

def model_chat(query, history=None,system='You are a helpful assistant.') :
    """
    生成器输出流结果
    :param query:
    :param history:
    :param system:
    :return:
    """
    if query is None:
        query = ''
    if history is None:
        history = []

    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    prompt = message_to_prompt(messages)

    gen_cfg = copy.deepcopy(engine_helper.default_gen_cfg)
    gen_cfg["seed"] = random.randint(0, 10000)

    request_list = engine_helper.create_request([prompt], [gen_cfg])

    request = request_list[0]
    # request_list[0].out_text
    gen = engine_helper.process_one_request_stream(request)
    return gen

def model_chat_all(query, history=None,system='You are a helpful assistant.') :
    """
    直接return结果，也是需要结果输出完成后才能打印
    :param query:
    :param history:
    :param system:
    :return:
    """
    if query is None:
        query = ''
    if history is None:
        history = []

    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    prompt = message_to_prompt(messages)

    gen_cfg = copy.deepcopy(engine_helper.default_gen_cfg)
    gen_cfg["seed"] = random.randint(0, 10000)

    request_list = engine_helper.create_request([prompt], [gen_cfg])

    request = request_list[0]
    # request_list[0].out_text
    gen = engine_helper.process_one_request_stream(request)
    # 需要从生成器中取数据才能有结果
    responses = [i for i in gen if i]  # 不取数据不会返回结果
    result = request_list[0].out_text
    return result

if __name__ == '__main__':
    from IPython.display import display, clear_output

    input_value = '写个四字成语'
    gen = model_chat(input_value)
    for part in gen: # 流式查看数据
        clear_output(wait=True)
        print(f"Input: {input_value}")
        print(f"Response:\n{part}")