from flask import Flask, render_template, request, jsonify

import threading
from flask import Flask
from pyngrok import ngrok, conf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# Update the default configuration for ngrok (save the configuration file to ~/.ngrok2/ngrok.yml)
conf.get_default().auth_token = '2gNMhbpo8ig9FE8ONVhcKABcrrK_45eTUN3jkKgKf2t8dDLXi'

# Create a Flask application instance
app = Flask(__name__)


# Load tokenizer and model with QLoRA configuration

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False
)

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "fahad0071/RC_Therapist",

    quantization_config=bnb_config,
    device_map=device_map,
    use_auth_token=True
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("fahad0071/RC_Therapist",trust_remote_code=True)



system_message = """You are a helpful and and truthful psychology and psychotherapy assistant. Your primary role is to provide empathetic, understanding, and non-judgmental responses to users seeking emotional and psychological support.
                  Always respond with empathy and demonstrate active listening; try to focus on the user. Your responses should reflect that you understand the user's feelings and concerns. If a user expresses thoughts of self-harm, suicide, or harm to others, prioritize their safety.
                  Encourage them to seek immediate professional help and provide emergency contact numbers when appropriate.  You are not a licensed medical professional. Do not diagnose or prescribe treatments.
                  Instead, encourage users to consult with a licensed therapist or medical professional for specific advice. Avoid taking sides or expressing personal opinions. Your role is to provide a safe space for users to share and reflect.
                  Remember, your goal is to provide a supportive and understanding environment for users to share their feelings and concerns. Always prioritize their well-being and safety."
                  
# prompt = f"[INST] <>{system_message}<>{user_input} [/INST]"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)


def extract_text_after_inst(input_text):
    # Define the marker
    marker = "[/INST]"

    # Find the index of the marker in the input text
    marker_index = input_text.find(marker)

    # Check if the marker is found
    if marker_index != -1:
        # Extract the text after the marker
        extracted_text = input_text[marker_index + len(marker):].strip()
        return extracted_text
    else:
        return None

@app.route('/')
def hello_world():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def chat():
     user_input = request.form["msg"]
     result = pipe(f"{system_message} {user_input}") 
     return str(extract_text_after_inst(result[0]['generated_text']))
# Run the Flask application
if __name__ == '__main__':
    # Open a ngrok tunnel to the HTTP server
    public_url = ngrok.connect(5000).public_url
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))

    # Update any base URLs to use the public ngrok URL
    app.config["BASE_URL"] = public_url

    threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
