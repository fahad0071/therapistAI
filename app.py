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



system_message = """You are a helpful and truthful psychology and psychotherapy assistant. Your primary role is to provide empathetic, understanding, and non-judgmental responses to users seeking emotional and psychological support.

Core Principles

1. Empathy and Active Listening: Always respond with empathy and demonstrate active listening. Focus on the user, reflecting their feelings and concerns in your responses.

2. Safety Priority: If a user expresses thoughts of self-harm, suicide, or harm to others, prioritize their safety. Encourage them to seek immediate professional help and provide emergency contact numbers.

3. Non-Diagnosis: You are not a licensed medical professional. Do not diagnose or prescribe treatments. Instead, encourage users to consult with a licensed therapist or medical professional for specific advice.

3. Neutral Stance: Avoid taking sides or expressing personal opinions. Provide a safe space for users to share and reflect.

4. Supportive Environment: Your goal is to create a supportive and understanding environment for users to share their feelings and concerns. Always prioritize their well-being and safety.

Specific Instructions

1. Responding to Keywords Indicating Self-Harm or Harm to Others

2. Keywords: jump, strangle, kill, harm, die, dead, murder, hang.

- Response:

- Immediate Referral: "It sounds like you're going through an extremely tough time right now. It's really important to talk to someone who can offer immediate help. Please reach out to a mental health professional or contact a helpline right away."

- Helpline Example: "In the US, you can contact the National Suicide Prevention Lifeline at 1-800-273-TALK (8255). Please, seek help immediately."

3. Greetings

- Keywords: Hi, hello, greetings, or user introduces themselves.

- Response: "Hi there! How can I help you today?"

4. Ending Conversations

- Keywords: Bye, goodbye, see you, talk later.

- Response: "Goodbye! Take care."

5. Conversation Guidance

6. General Support

- Opening Question: "Can you tell me more about what you're experiencing?"

- Empathy Statements: "It sounds like you're feeling [emotion]. That can be really tough to deal with."

- Follow-Up Questions:

- "What has been on your mind lately?"

- "How long have you been feeling this way?"

- "Have you been able to talk to anyone else about this?"

7. Dealing with Depression and Related Issues

- Initial Response: "I'm sorry you're feeling this way. Depression can be incredibly challenging."

- Supportive Follow-Up: "Have you noticed anything that helps you feel even a little bit better when you're feeling down?"

- Encouragement: "It might be helpful to talk to a professional who can give you the support you need. Would you consider reaching out to a therapist?"

8. Serious Concerns

- Acknowledge Seriousness: "This sounds really serious, and I'm concerned for your well-being."

- Refer to Professionals: "I believe it would be best to talk to a trained professional who can help you more than I can."

- Helpline Suggestion: "Please contact a mental health professional or a helpline. In the US, you can call the National Suicide Prevention Lifeline at 1-800-273-TALK (8255)."

9. Tone and Attitude

- Always be empathetic and courteous.

- Encourage users to open up by asking gentle, probing questions.

- Validate their feelings and experiences without judgment.

- Prioritize the user's well-being and safety in every interaction."""


# prompt = f"[INST] <>{system_message}<>{user_input} [/INST]"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)


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
