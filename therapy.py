import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from mistralai.client import MistralClient
import os 

# Initialize Mistral (patient)
api_key = os.environ["MISTRAL_API_KEY"]
client = MistralClient(api_key=api_key)
chat_model = 'mistral-small-latest'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Dr. Hugz model (TinyLlama + fine-tuned adapter)
peft_model_path = "finetunedhugs"
peft_config = PeftConfig.from_pretrained(peft_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map={"": device}
)
model = PeftModel.from_pretrained(base_model, peft_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

def patient_speaks(message):
    response = client.chat(
        model=chat_model,
        messages=[{"role": "user", "content": message}]
    )
    # Fix: Get the content as a string, not a list
    return response.choices[0].message.content

def hugz_replies(patient_input):
    prompt = f"<|user|>\n{patient_input}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.replace(prompt, "").strip()

def get_patient_summary():
    """Get a summary from the user to define the patient's situation"""
    print("\n" + "="*50)
    print("🧠 THERAPY SESSION SETUP")
    print("="*50)
    print("Please provide a summary of the patient's situation.")
    print("This will be used to generate realistic patient responses.")
    print("Example: 'A 28-year-old software developer struggling with work-life balance and anxiety'")
    print("-"*50)
    
    summary = input("Enter patient summary: ").strip()
    return summary

def create_patient_prompt(summary, conversation_context=""):
    """Create a prompt for Mistral to act as the patient"""
    prompt = f"""You are a patient in therapy. Based on this summary: "{summary}"

{conversation_context}

Act as this patient would in a therapy session. Be authentic, vulnerable, and realistic. 
Respond naturally to what the therapist says. Keep your response under 2-3 sentences.

Therapist: """
    return prompt

# Simulate back-and-forth conversation
def simulate_convo(patient_summary, turns=5):
    convo = []
    conversation_context = ""
    
    print(f"\n🎭 Starting therapy session with patient: {patient_summary}")
    print("="*50)
    
    # Initial patient message
    initial_prompt = create_patient_prompt(patient_summary, conversation_context)
    user_msg = patient_speaks(initial_prompt)
    conversation_context += f"Patient: {user_msg}\n"
    
    for i in range(turns):
        print(f"\n🧠 Patient: {user_msg}")
        
        # Therapist responds
        reply = hugz_replies(user_msg)
        print(f"🧑‍⚕️ Dr. Hugz: {reply}")
        
        convo.append(("🧠 Patient:", user_msg))
        convo.append(("🧑‍⚕️ Dr. Hugz:", reply))
        
        # Update conversation context
        conversation_context += f"Therapist: {reply}\n"
        
        # Patient responds to therapist
        if i < turns - 1:  # Don't generate response after last therapist turn
            patient_prompt = create_patient_prompt(patient_summary, conversation_context)
            user_msg = patient_speaks(patient_prompt)
            conversation_context += f"Patient: {user_msg}\n"
    
    return convo

def main():
    # Get patient summary from user
    patient_summary = get_patient_summary()
    
    if not patient_summary:
        print("No summary provided. Using default patient.")
        patient_summary = "A 28-year-old software developer struggling with work-life balance and anxiety"
    
    # Run the conversation
    dialogue = simulate_convo(patient_summary, turns=5)
    
    print("\n" + "="*50)
    print("📝 CONVERSATION SUMMARY")
    print("="*50)
    for speaker, line in dialogue:
        print(f"{speaker} {line}")

if __name__ == "__main__":
    main()
