import sys
import os
import re
import json
import base64
import math
from io import BytesIO
from PIL import Image
import argparse
from inference_engine.safe_persis_shared_vis_python_exe import PythonExecutor, ImageRuntime
from openai import OpenAI, BadRequestError
import anthropic

def encode_image(image):
    """
    Convert a PIL.Image object or image file path to base64-encoded string, and get resolution info.
    
    Args:
        image: Can be a PIL.Image object or image file path.
    Returns:
        dict with keys:
        - 'base64': base64-encoded string
        - 'width': width in pixels
        - 'height': height in pixels
        - 'resolution': string "widthxheight"
    """
    img_obj = None
    
    if isinstance(image, str):
        # Handle file path
        img_obj = Image.open(image)
        with open(image, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Handle PIL.Image object
        img_obj = image
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    width, height = img_obj.size
    
    return {
        'base64': base64_str,
        'width': width,
        'height': height
    }

def encode_image_with_resize(image):
    """
    Convert a PIL.Image object or image file path to base64-encoded string, get resolution info.
    If resolution > 1024x1024, resize to half.
    
    Args:
        image: Can be a PIL.Image object or image file path
    Returns:
        dict with keys:
        - 'base64': base64-encoded string
        - 'width': width in pixels
        - 'height': height in pixels
        - 'resolution': string "widthxheight"
    """
    img_obj = None
    
    if isinstance(image, str):
        img_obj = Image.open(image)
    else:
        img_obj = image

    # Resize if larger than 1024x1024
    width, height = img_obj.size
    if width > 1024 or height > 1024:
        new_size = (width // 2, height // 2)
        img_obj = img_obj.resize(new_size, Image.LANCZOS)
        width, height = img_obj.size

    buffered = BytesIO()
    img_obj.save(buffered, format='PNG')
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        'base64': base64_str,
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}"
    }

def check(evaluator, pred_ans, real_ans):
    if len(pred_ans) == 0:
        return []
    correctness = evaluator.score(pred_ans, real_ans)
    return correctness

def execute_codes(codes, messages, executor: PythonExecutor,image_path=None):
    no_code_idx = []
    codes_use = []
    for i, code in enumerate(codes):
        if code == "":
            no_code_idx.append(i)
        else:
            codes_use.append(code)
    # Include the image path argument
    batch_results = executor.batch_apply(codes_use, messages,input_image_path=image_path)
    return batch_results, no_code_idx

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def round_by_factor(number, factor):
    return round(number / factor) * factor

def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor

def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor

def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def _strip_json_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

VISUAL_TOOL_PROMPTS = {"deepeyes", "pixelreasoner"}


def is_visual_tool_prompt(prompt_name):
    return prompt_name in VISUAL_TOOL_PROMPTS


def extract_visual_tool_call(response_text):
    if "<tool_call>" in response_text:
        tool_text = response_text.split("<tool_call>", 1)[1]
        if "</tool_call>" in tool_text:
            tool_text = tool_text.split("</tool_call>", 1)[0]
    else:
        tool_text = response_text
    return json.loads(_strip_json_fence(tool_text))


def _bbox_to_pixel_box(bbox, width, height, normalized=False):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if normalized:
        left = max(0, min(width, int(round(x1 * width))))
        top = max(0, min(height, int(round(y1 * height))))
        right = max(0, min(width, int(round(x2 * width))))
        bottom = max(0, min(height, int(round(y2 * height))))
    else:
        left = max(0, min(width, int(round(x1))))
        top = max(0, min(height, int(round(y1))))
        right = max(0, min(width, int(round(x2))))
        bottom = max(0, min(height, int(round(y2))))
    return left, top, right, bottom


def execute_visual_zoom_tool(response_text, tool_images):
    tool_call = extract_visual_tool_call(response_text)
    tool_name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", tool_call)
    bbox = arguments.get("bbox_2d") or arguments.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid visual bbox tool call: {tool_call}")

    target_image = arguments.get("target_image")
    if target_image is None:
        image_index = len(tool_images) - 1
    else:
        image_index = int(target_image) - 1

    if image_index < 0 or image_index >= len(tool_images):
        raise ValueError(f"target_image={target_image} is out of range for {len(tool_images)} available images")

    image = tool_images[image_index].copy().convert("RGB")
    width, height = image.size
    normalized = tool_name == "crop_image_normalized" or all(0.0 <= float(v) <= 1.0 for v in bbox)
    left, top, right, bottom = _bbox_to_pixel_box(bbox, width, height, normalized=normalized)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid visual bbox after clipping: {bbox} for image size {width}x{height}")

    cropped_image = image.crop((left, top, right, bottom))
    new_h, new_w = smart_resize(cropped_image.height, cropped_image.width)
    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

    buffered = BytesIO()
    cropped_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    text_result = (
        f"{tool_name or 'visual_tool'} executed. "
        f"target_image={image_index + 1}, bbox_2d={[left, top, right, bottom]}"
    )
    return [image_base64], text_result, cropped_image


def process_prompt_init(question, image_path_list, prompt_template, prompt_type, api_name):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]

    image_path = image_path_list[0]

    if "<IMAGE_PLACE_HOLDER_0>" in question:
        if "no_tool" in prompt_type:

            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            question_with_options = question
            # Include the image path argument
            question = prompt_prefix.format(query=question_with_options,image_path=image_path)

            parts = question.split("<IMAGE_PLACE_HOLDER_0>")
            content = []
            
            # Add text before image (if any)
            if parts[0].strip():
                content.append({"type": "text", "text": parts[0].strip()})
            # Add image
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            
            # Add text after image (if any)
            if len(parts) > 1 and parts[1].strip():
                content.append({"type": "text", "text": parts[1].strip()})

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            return messages

        else:
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            width = img_result['width']
            height = img_result['height']
            question_with_options = question
            # Include the image path argument
            question = prompt_prefix.format(query=question_with_options, width=str(width), height=str(height),image_path=image_path)

            # Split question into parts
            parts = question.split("<IMAGE_PLACE_HOLDER_0>")
            # Build message with image_clue tags
            content = []
            
            # Add text before image (if any)
            if parts[0].strip():
                content.append({"type": "text", "text": parts[0].strip()})
            
            # Add image with tags
            content.append({"type": "text", "text": "<image_clue_0>"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            content.append({"type": "text", "text": "</image_clue_0>\n\n"})
            
            # Add text after image (if any)
            if len(parts) > 1 and parts[1].strip():
                content.append({"type": "text", "text": parts[1].strip()})

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            return messages

    else:
        if "no_tool" in prompt_type:

            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            question_with_options = question

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": prompt_prefix.format(query=question_with_options,image_path=image_path)}]
                }
            ]

            return messages

        else:
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            width = img_result['width']
            height = img_result['height']
            question_with_options = question

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "<image_clue_0>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": "</image_clue_0>\n\n"}] + [{"type": "text", "text": prompt_prefix.format(query=question_with_options, width=str(width), height=str(height),image_path=image_path)}]
                }
            ]

            return messages

def process_prompt_init_multi_images(question, image_path_list, prompt_template, prompt_type, api_name):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]
    
    # Prepare image data
    image_data = []
    image_information = ""
    
    for i, image_path in enumerate(image_path_list):
        if "claude" in api_name:
            img_result = encode_image_with_resize(image_path)
        else:
            img_result = encode_image(image_path)
        image_base64 = img_result['base64']
        width = img_result['width']
        height = img_result['height']
        
        image_data.append({
            "index": i,
            "base64": image_base64,
            "width": width,
            "height": height,
            "placeholder": f"<IMAGE_PLACE_HOLDER_{i}>"
        })
        
        image_information += f"width of image_clue_{i}: {width}, height of image_clue_{i}: {height}\n"
    
    # Format question
    # Prepare additional arguments that might be needed by some templates (e.g., 'thyme')
    first_width = image_data[0]['width'] if image_data else "N/A"
    first_height = image_data[0]['height'] if image_data else "N/A"
    image_paths_str = str(image_path_list)

    formatted_question = prompt_prefix.format(
        query=question, 
        image_information=image_information,
        image_path=image_paths_str,
        width=str(first_width),
        height=str(first_height)
    )
    
    # Check if placeholder exists
    has_placeholders = any(f"<IMAGE_PLACE_HOLDER_{i}>" in formatted_question for i in range(len(image_path_list)))
    
    if has_placeholders:
        # Insert images at placeholder positions
        if "no_tool" in prompt_type:
            content = []
            remaining_text = formatted_question
            
            for img_data in image_data:
                placeholder = img_data["placeholder"]
                if placeholder in remaining_text:
                    parts = remaining_text.split(placeholder, 1)
                    
                    if parts[0]:
                        content.append({"type": "text", "text": parts[0]})
                    
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                    
                    remaining_text = parts[1]
            
            if remaining_text:
                content.append({"type": "text", "text": remaining_text})
            
            messages = [{"role": "user", "content": content}]
            return messages
        else:
            content = []
            remaining_text = formatted_question
            
            for img_data in image_data:
                placeholder = img_data["placeholder"]
                if placeholder in remaining_text:
                    parts = remaining_text.split(placeholder, 1)
                    
                    if parts[0]:
                        content.append({"type": "text", "text": parts[0]})
                    
                    i = img_data["index"]
                    content.append({"type": "text", "text": f"<image_clue_{i}>"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                    content.append({"type": "text", "text": f"</image_clue_{i}>\n\n"})
                    
                    remaining_text = parts[1]
            
            if remaining_text:
                content.append({"type": "text", "text": remaining_text})
            
            messages = [{"role": "user", "content": content}]
            return messages
    else:
        # Handle as usual if no placeholder
        if "no_tool" in prompt_type:
            content = []
            
            for i, img_data in enumerate(image_data):
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
            
            content.append({"type": "text", "text": formatted_question})
            
            messages = [{"role": "user", "content": content}]
            return messages
        else:
            content = []
            
            for i, img_data in enumerate(image_data):
                content.append({"type": "text", "text": f"<image_clue_{i}>"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                content.append({"type": "text", "text": f"</image_clue_{i}>\n\n"})
            
            content.append({"type": "text", "text": formatted_question})
            
            messages = [{"role": "user", "content": content}]
            return messages


def update_messages_with_execute_content(image_nums_in_input, messages, images_result, text_result, error_result, image_clue_idx):
    if error_result is None:
        new_messages = []
        image_content = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)

        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>\nText Result:\n{text_result}\nImage Result:\n"}]
        if images_result is not None:
            print(f"#### image_clue_index: {image_clue_idx},Image_nums_in_input: {image_nums_in_input}, len of images_result: {len(images_result)}")
            # for image_base64_item in images_result[image_clue_idx-image_nums_in_input:]:
            for image_base64_item in images_result:
                interpreter_message_images = [{"type": "text", "text": f"<image_clue_{image_clue_idx}>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_item}"}}] + [{"type": "text", "text": f"</image_clue_{image_clue_idx}>"}]
                image_content += interpreter_message_images
                image_clue_idx += 1
        else:
            image_content = [{"type": "text", "text": "None"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]

        interpreter_message_item = interpreter_message_text_prefix + image_content + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})
        new_messages.append({"role": "user", "content": interpreter_message_item})
    else:
        new_messages = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)
    
        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>{error_result}"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]
    
        interpreter_message_item = interpreter_message_text_prefix + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})
        new_messages.append({"role": "user", "content": interpreter_message_item})

    return new_messages, image_clue_idx

def update_messages_with_code(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}</code>\n"}]
    }

    messages.append(message_item)
    return messages

def update_messages_with_tool_call(messages, generated_content):
    if "<tool_call>" in generated_content and "</tool_call>" not in generated_content:
        generated_content = f"{generated_content}</tool_call>"
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}\n"}]
    }

    messages.append(message_item)
    return messages

def update_messages_with_text(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}"}]
    }

    messages.append(message_item)
    return messages

def call_chatgpt_api(args, messages, client, max_tokens=10000, stop=None, temperature=0.6):
    """Call ChatGPT API with the given messages"""
    try:
        client_type = args.client_type
        api_name = args.api_name
    except:
        client_type = args['client_type']
        api_name = args['api_name']
    
    # Track input and output token counts
    usage = {"input": 0, "output": 0}


    if client_type == "openai" or client_type == "azure":
        
        try:
            response = client.chat.completions.create(
                model=api_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=stop,
                timeout=300
            )

            # Read input and output token counts
            response_text = response.choices[0].message.content
            try:
                usage["input"] = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                usage["output"] = int(getattr(response.usage, "completion_tokens", 0) or 0)
            except:
                pass

        except BadRequestError as e:
            if "stop" in str(e) and stop is not None:
                # Retry without stop parameter if it's not supported
                response = client.chat.completions.create(
                    model=api_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1.0,
                    # stop=stop,
                    timeout=300
                )

                # Read input and output token counts
                response_text = response.choices[0].message.content
                try:
                    usage["input"] = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                    usage["output"] = int(getattr(response.usage, "completion_tokens", 0) or 0)
                except:
                    pass

            else:
                raise e
        response_text = response.choices[0].message.content
    elif client_type == "anthropic":
        message = client.messages.create(
            model=api_name,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
            top_p=1.0,
            stop_sequences=stop
        )  
        response_text = message.content[0].text if isinstance(message.content, list) else message.content
    elif client_type == "vllm":
        response = client.chat.completions.create(
            model=api_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=stop
        )
        # Read input and output token counts
        response_text = response.choices[0].message.content
        try:
            usage["input"] = int(getattr(response.usage, "prompt_tokens", 0) or 0)
            usage["output"] = int(getattr(response.usage, "completion_tokens", 0) or 0)
        except:
            pass
    else:
        print("Your args.client_type must be one of openai, azure, anthropic and vllm.")
        return None, None

    if stop and "</answer>" in stop and "<answer>" in response_text and "</answer>" not in response_text:
        response_text += "</answer>"
    
    # Check if stop sequence is encountered
    stop_reason = None
    if stop and any(s in response_text for s in stop):
        for s in stop:
            if s in response_text:
                stop_reason = s
                break
    else:
        if client_type in ["openai", "azure", "vllm"]:
            stop_reason = response.choices[0].finish_reason
        else:
            stop_reason = "stop"

    if "<tool_call>" in response_text:
        stop_reason = "</tool_call>"
    if "<code>" in response_text:
        stop_reason = "</code>"
    
    return response_text, stop_reason,usage

def evaluate_single_data(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
        temperature = args.temperature
        api_name = args.api_name
        max_rounds = getattr(args, "max_rounds", 3)
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
        temperature = args['temperature']
        api_name = args['api_name']
        max_rounds = args.get('max_rounds', 3)

    image_path_list = data['image_path_list']

    if "no_tool" in prompt:
        if len(image_path_list) == 1:
            messages = process_prompt_init(data["question"], image_path_list, prompt_template, prompt, api_name)
        elif len(image_path_list) >= 2:
            messages = process_prompt_init_multi_images(data["question"], image_path_list, prompt_template, prompt, api_name)
    else:
        if len(image_path_list) == 1:
            messages = process_prompt_init(data["question"], image_path_list, prompt_template, prompt, api_name)
        elif len(image_path_list) >= 2:
            messages = process_prompt_init_multi_images(data["question"], image_path_list, prompt_template, prompt, api_name)
    
    # count token
    total_input_tokens = 0
    total_output_tokens = 0

    # Check for forced action sequence execution (Slow Path)
    forced_action_mode = False
    if "action_seq" in data and "map_desc" in data:
        forced_action_mode = True

    if forced_action_mode:
        # Construct the python code to render the map
        # map_desc is expected to be a string representation of a list, e.g. "['FFF', ...]"
        map_desc_str = data["map_desc"]
        action_seq_str = data["action_seq"]
        start_state = data.get("start_state", None)
        
        # Use dedent or explicit formatting to ensure no indentation in the generated string
        code = f"""map_desc_str = "{map_desc_str}"
actions = "{action_seq_str}"
start_state = {start_state}
run_maze(map_desc_str, actions, start_state)"""
        
        response_text = f"Executing action plan...<code>\n```python\n{code}\n```\n</code>"
        pred_stop_reason = "</code>"
        usage = {"input": 0, "output": 0}
    else:
        if is_visual_tool_prompt(prompt) and exe_code:
            stop = ["</tool_call>", "</answer>"]
        elif exe_code:
            stop = ["</code>"]
        else:
            stop = None
        # Generate initial response normally
        response_text, pred_stop_reason, usage = call_chatgpt_api(
            args,
            messages, 
            client,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature
        )
    
    total_input_tokens += usage.get("input", 0)
    total_output_tokens += usage.get("output", 0)

    # Handle response
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = len(image_path_list)
    tool_images = [Image.open(image_path).convert("RGB") for image_path in image_path_list]
    
    while True:
        if is_visual_tool_prompt(prompt) and exe_code and pred_stop_reason == "</tool_call>":
            if max_rounds is not None and code_execution_count >= int(max_rounds):
                messages = update_messages_with_tool_call(messages, response_text)
                response_text, pred_stop_reason, usage = call_chatgpt_api(
                    args,
                    messages,
                    client,
                    max_tokens=max_tokens,
                    stop=["</answer>"],
                    temperature=temperature
                )
                total_input_tokens += usage.get("input", 0)
                total_output_tokens += usage.get("output", 0)
                final_response = response_text
                messages = update_messages_with_text(messages, response_text)
                break

            messages = update_messages_with_tool_call(messages, response_text)
            try:
                images_result, text_result, cropped_image = execute_visual_zoom_tool(response_text, tool_images)
                tool_images.append(cropped_image)
                error_result = None
            except Exception as e:
                images_result = None
                text_result = None
                error_result = f"Visual tool execution error: {e}"

            messages, new_image_clue_idx = update_messages_with_execute_content(
                len(image_path_list), messages, images_result, text_result, error_result, image_clue_idx
            )
            image_clue_idx = new_image_clue_idx
            code_execution_count += 1

            if max_rounds is not None and code_execution_count >= int(max_rounds):
                response_text, pred_stop_reason, usage = call_chatgpt_api(
                    args,
                    messages,
                    client,
                    max_tokens=max_tokens,
                    stop=["</answer>"],
                    temperature=temperature
                )
                total_input_tokens += usage.get("input", 0)
                total_output_tokens += usage.get("output", 0)
                final_response = response_text
                messages = update_messages_with_text(messages, response_text)
                break

            response_text, pred_stop_reason, usage = call_chatgpt_api(
                args,
                messages,
                client,
                max_tokens=max_tokens,
                stop=["</tool_call>", "</answer>"],
                temperature=temperature
            )
            total_input_tokens += usage.get("input", 0)
            total_output_tokens += usage.get("output", 0)

        # Check if code execution is needed
        elif exe_code and not is_visual_tool_prompt(prompt) and pred_stop_reason == "</code>":
            if max_rounds is not None and code_execution_count >= int(max_rounds):
                # Max rounds reached before execution: make one final API call
                # without </code> stop to let the model produce a concluding answer
                messages = update_messages_with_code(messages, response_text)
                response_text, pred_stop_reason, usage = call_chatgpt_api(
                    args,
                    messages,
                    client,
                    max_tokens=max_tokens,
                    stop=["</answer>"],
                    temperature=temperature
                )
                total_input_tokens += usage.get("input", 0)
                total_output_tokens += usage.get("output", 0)
                final_response = response_text
                messages = update_messages_with_text(messages, response_text)
                break
            # Extract code to execute
            messages = update_messages_with_code(messages, response_text)

            # Prefer Python code inside <code> tags
            if "<code>" in response_text and "</code>" in response_text:
                code_content = response_text.split("<code>")[-1].split("</code>")[0]
                if "```python" in code_content:
                    code_to_execute = code_content.split("```python")[-1].split("```")[0].strip()
                else:
                    code_to_execute = code_content.strip()
                    if code_to_execute.startswith("python"):
                        code_to_execute = code_to_execute[6:].strip()
            else:
                # Fall back to the original extraction logic
                code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # Get the image path
            current_image_path = image_path_list[0] if image_path_list and len(image_path_list) > 0 else None

            # Execute code
            exe_result = execute_codes([code_to_execute], messages, executor,image_path=current_image_path)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_execute_content(len(image_path_list), messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            if max_rounds is not None and code_execution_count >= int(max_rounds):
                # Max rounds reached: make one final API call without </code> stop
                # to let the model produce a concluding answer
                response_text, pred_stop_reason, usage = call_chatgpt_api(
                    args,
                    messages,
                    client,
                    max_tokens=max_tokens,
                    stop=["</answer>"],
                    temperature=temperature
                )
                total_input_tokens += usage.get("input", 0)
                total_output_tokens += usage.get("output", 0)
                final_response = response_text
                messages = update_messages_with_text(messages, response_text)
                break
            
            # Generate next response part
            response_text, pred_stop_reason, usage= call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>", "</answer>"] if exe_code else ["</answer>"],
                temperature=temperature
            )

            total_input_tokens += usage.get("input", 0)
            total_output_tokens += usage.get("output", 0)

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response, total_input_tokens, total_output_tokens


def evaluate_single_data_multi_images(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
        api_name = args.api_name
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
        api_name = args['api_name']

    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt, api_name)
    
    # Generate initial response
    response_text, pred_stop_reason, _ = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # Handle response
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # Check if code execution is needed
        if exe_code and pred_stop_reason == "</code>":
            # Extract code to execute
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # Execute code
            exe_result = execute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_execute_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # Generate next response part
            response_text, pred_stop_reason, _ = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response

def evaluate_single_data_video(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
        api_name = args.api_name
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
        api_name = args['api_name']

    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt, api_name)
    
    # Generate initial response
    response_text, pred_stop_reason, _ = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # Handle response
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # Check if code execution is needed
        if exe_code and pred_stop_reason == "</code>":
            # Extract code to execute
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # Execute code
            exe_result = execute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_execute_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # Generate next response part
            response_text, pred_stop_reason, _ = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response


# New wrapper functions for safe execution with cleanup
def evaluate_batch_with_cleanup(args, data_list, client):
    """Wrapper function to ensure proper cleanup of resources when processing multiple items"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data(args, data, client, executor)
                results.append(result)
            except Exception as e:
                print(f"Error processing data item: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor

def evaluate_single_with_cleanup(args, data, client):
    """Wrapper function for evaluating a single item with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)

    try:
        result = evaluate_single_data(args, data, client, executor)
        return result
    finally:
        # Ensure cleanup of persistent worker
        del executor

def evaluate_multi_images_with_cleanup(args, data_list, client):
    """Wrapper function for multi-image evaluation with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data_multi_images(args, data, client, executor)
                results.append(result)
            except Exception as e:
                print(f"Error processing multi-image data: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor

def evaluate_video_with_cleanup(args, data_list, client):
    """Wrapper function for video evaluation with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data_video(args, data, client, executor)
                results.append(result)
            except Exception as e:
                print(f"Error processing video data: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor
