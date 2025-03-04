import gradio as gr
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "YuchengShi/LLaVA-v1.5-7B-Stanford-Dogs"
processor = AutoProcessor.from_pretrained(model_id)

# We'll load model in a function to control when GPU memory is allocated
def load_model():
    global model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    return "Model loaded successfully!"

def predict(image, question="What breed is this dog?"):
    if model is None:
        return "Error: Model not loaded. Please click 'Load Model' first."
    
    # Create conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        },
    ]
    
    # Process inputs
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, dtype)
    
    # Generate prediction
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    prediction = processor.decode(output[0][2:], skip_special_tokens=True)
    
    return prediction

# Initialize model to None - will load on button click
model = None

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Dog Breed Recognition with LLaVA")
    gr.Markdown("Upload a dog image to identify the breed. First click 'Load Model' (this will take a minute), then upload your image.")
    
    with gr.Row():
        with gr.Column():
            load_button = gr.Button("Load Model")
            image_input = gr.Image(type="pil")
            question_input = gr.Textbox(
                label="Question (default is 'What breed is this dog?')", 
                value="What breed is this dog?",
            )
            submit_btn = gr.Button("Identify Breed")
        
        with gr.Column():
            output_text = gr.Textbox(label="Result")
    
    load_button.click(load_model, inputs=None, outputs=gr.Textbox())
    submit_btn.click(predict, inputs=[image_input, question_input], outputs=output_text)

demo.launch()
