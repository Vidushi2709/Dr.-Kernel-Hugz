import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from mistralai.client import MistralClient

# Initialize Mistral (patient)
client = MistralClient(api_key="NIbuIX8xXFyKFtz6fUJrqQjEabdbmSP6")
chat_model = 'mistral-small-latest'

# Load Dr. Hugz model (TinyLlama + fine-tuned adapter)
peft_model_path = "finetunedhugs"
peft_config = PeftConfig.from_pretrained(peft_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, peft_model_path).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

def patient_speaks(message):
    response = client.chat(
        model=chat_model,
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content.strip()

def hugz_replies(patient_input):
    prompt = f"<|user|>\n{patient_input}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.replace(prompt, "").strip()

# Simulate back-and-forth conversation
def simulate_convo(initial_prompt, turns=5):
    convo = []
    user_msg = patient_speaks(initial_prompt)

    for i in range(turns):
        convo.append(("🧠 AI:", user_msg))
        reply = hugz_replies(user_msg)
        convo.append(("🧑‍⚕️ Dr. Hugz:", reply))
    return convo

# Initial message from Mistral
starting_prompt = ''' 🧠 Therapy Roleplay Setup – Mistral Style
You kicked off the day with a bold idea: simulate a conversation between a patient and a therapist, using the base Mistral model as the patient and a fine-tuned model as the therapist. You wanted to control the convo based on a summary you provide, limiting it to 5 back-and-forths. We talked about how to implement this with the Mistral API and proper prompt flow.

📸 Interface Image Shared
You showed me what you were trying to build — likely a UI or screenshot mockup — and we discussed how to connect that with model inputs/outputs for clean turn-taking between patient and therapist.

🧪 Study Buddy for Interview Prep
Somewhere between model wrangling and therapeutic roleplay, you pivoted to interview prep. You used me to review key concepts (like SMOTE, class imbalance, and model parameters), and even wrote out explanations like you'd give in a real interview.
(You were nervous — totally valid — but you also nailed your answers.)

'''

# Run the convo
dialogue = simulate_convo(starting_prompt, turns=5)
for speaker, line in dialogue:
    print(f"{speaker}\n{line}\n")
