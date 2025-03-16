import logging
import json
from vllm import LLM, SamplingParams
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-72B-Instruct-AWQ")

sampling_params = SamplingParams(temperature=0.1, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

llm = LLM(model="Qwen2.5-72B-Instruct-AWQ", gpu_memory_utilization=0.8, tensor_parallel_size=2, max_model_length=28496, quantization="awq")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")
Settings.transformations = [SentenceSplitter(chunk_size=1024)]

def load_cantonese_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading dictionary file: {e}")
        return {}

def build_index(cantonese_dict):
    try:
        documents = [{"text": value, "metadata": {"key": key}} for key, value in cantonese_dict.items()]
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=Settings.embed_model,
            transformations=Settings.transformations
        )
        return index
    except Exception as e:
        logging.error(f"Error building index: {e}")
        return None

def save_index(index, save_path="index_storage"):
    storage_context = StorageContext.from_defaults(persist_dir=save_path)
    index.storage_context = storage_context
    index.storage_context.persist()

def load_index(save_path="index_storage"):
    storage_context = StorageContext.from_defaults(persist_dir=save_path)
    return load_index_from_storage(storage_context)

def generate_cantonese_response(scene_description, index):
    if not index:
        logging.error("Index not loaded. Cannot generate response.")
        return ""

    try:
        query_engine = index.as_query_engine()
        query_result = query_engine.query(scene_description)

        retrieved_text = query_result.response if query_result.response else "No relevant content retrieved. Generating based on general knowledge:"

        content = """
                ## Role: Cantonese Text Generator

                ## Profile:
                - author: Language Model
                - version: 0.1
                - language: Cantonese
                - description: I can generate Cantonese expressions based on given scenarios or instructions.

                ## Goals:
                - Generate Cantonese dialogues or descriptions based on user-provided scenarios or instructions, adhering to Cantonese language conventions.

                ## Constraints:
                - Avoid unnecessary English.
                - The output text should conform to Cantonese expression styles and conventions.
                - No grammatical errors.
                - The generated text should be as relevant as possible to the provided scenario.
                - Output format: <\Cantonese>Cantonese content<\Cantonese>

                ## Examples:
                [input]: 午后，和朋友闲聊吃饭的内容，谈到了最近尝试的餐厅和菜品。
                [answer]: <\Cantonese>今日午後，我同埋几个朋友坐喺咖啡店度，大家一边饮咖啡一边闲聊。我话我最近去咗一间新开嘅餐厅，食咗个招牌菜——香煎三文鱼，鱼肉好鲜嫩，外皮煎到金黄，入口即化，配埋个柠檬汁，清新又唔腻。朋友就话佢去咗一个老街边嘅小馆子，食咗个叉烧饭，叉烧甜中带咸，肉质软糯，饭粒粒分明，淋上个酱汁，好好食。我又讲，我之前试过一个素菜餐厅，食咗个麻婆豆腐，虽然冇肉，但豆腐嫩滑，辣得刚刚好，好有味道。大家你一言我一语，分享自己嘅美食经历，觉得好开心。<\Cantonese>

                [input]: 今天登山归来写游记
                [answer]: <\Cantonese>今日好天气，我同埋几个朋友一早就去爬山踏青。我们选择咗附近一个比较少人去嘅小山，想感受一下宁静嘅自然风光。一路上，阳光暖暖嘅，微风轻轻吹过，好舒服。山路边上开满咗野花，红红白白嘅，好靓。我们一边行一边聊天，心情好放松。行到半山腰，有个小溪流，溪水清清，哗啦啦地流过石头。我哋停下来，脱咗鞋，把脚浸入溪水里，冻凉冻凉嘅，好爽。继续往上爬，路越来越陡，大家都有点喘气，但都唔想放弃。终于爬到山顶，视野一下子开阔咗，四周嘅山峦连绵起伏，远处嘅城市看起来好小。我们找咗一块草地坐低，打开背包，开始野餐。有三文鱼三明治、水果、还有自己带嘅茶水，食得好开心。休息咗一阵，我们开始下山。下山嘅路比较容易行，大家一路咁说笑，时间过得好快。回到家，虽然有点累，但心情好好，觉得今日嘅踏青好充实，下次还要再找机会去爬山。<\Cantonese>
                """

        prompt = f"""
        {content}

        Retrieved content: {retrieved_text}
        Scene description: {scene_description}
        """
        response = llm.generate([prompt], sampling_params=sampling_params)
        return response[0].outputs[0].text
    except Exception as e:
        logging.error(f"Error generating Cantonese response: {e}")
        return ""

# 主程序
cantonese_dict = load_cantonese_dict("cantonese_dict.json")
index = build_index(cantonese_dict)
save_index(index)

scene_description = ""
response = generate_cantonese_response(scene_description, index)
logging.info(f"Generated Cantonese dialogue: {response}")