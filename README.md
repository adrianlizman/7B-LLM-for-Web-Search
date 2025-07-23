# Complete Step-by-Step Training Guide 

## 🚀 Overview
We'll train a 7B model for web search in these phases:
1. **Setup** (30 minutes)
2. **Generate Dataset** (2-3 hours)  
3. **Train Model** (4-6 hours)
4. **Deploy to Ollama** (30 minutes)
5. **Optimize Performance** (15 minutes)

---

## Phase 1: Environment Setup (30 minutes)

### Step 1.1: Install Python Dependencies
```bash
# Create a new conda environment (recommended)
conda create -n websearch-llm python=3.10
conda activate websearch-llm

# Install PyTorch with CUDA support for RTX 3060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install training dependencies
pip install transformers==4.36.0
pip install datasets==2.14.0
pip install accelerate==0.24.0
pip install bitsandbytes==0.41.3
pip install peft==0.6.0
pip install trl==0.7.4

# Install web search dependencies
pip install requests beautifulsoup4 duckduckgo-search

# Install monitoring tools
pip install GPUtil matplotlib wandb

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 1.2: Create Project Directory
```bash
mkdir ~/websearch-llm-training
cd ~/websearch-llm-training

# Create subdirectories
mkdir data models scripts logs
```

### Step 1.3: Test GPU Setup
Create file: `test_gpu.py`
```python
import torch
import transformers

print("🔥 GPU Setup Test")
print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} - {props.total_memory / 1e9:.1f}GB")

if torch.cuda.device_count() == 2:
    print("✅ Dual RTX 3060 setup detected!")
else:
    print("⚠️  Expected 2 GPUs, found", torch.cuda.device_count())
```

**Run it:**
```bash
python test_gpu.py
```

**Expected output:**
```
🔥 GPU Setup Test
PyTorch Version: 2.1.0+cu118
Transformers Version: 4.36.0
CUDA Available: True
Number of GPUs: 2
GPU 0: NVIDIA GeForce RTX 3060 - 12.0GB
GPU 1: NVIDIA GeForce RTX 3060 - 12.0GB
✅ Dual RTX 3060 setup detected!
```

---

## Phase 2: Generate Training Dataset (2-3 hours)

### Step 2.1: Create Dataset Generator
Create file: `generate_dataset.py`
```python
import json
import requests
from duckduckgo_search import DDGS
import time
import random
from tqdm import tqdm
import os

class WebSearchDatasetGenerator:
    def __init__(self):
        self.ddgs = DDGS()
        
    def generate_search_examples(self, num_examples=1000):
        """Generate training examples for web search"""
        
        # Expanded query templates for better diversity
        query_templates = [
            "What is {}?",
            "How to {}?", 
            "Latest news about {}",
            "Best {} 2024",
            "Tutorial for {}",
            "Compare {} vs {}",
            "History of {}",
            "Price of {}",
            "Review of {}",
            "Explain {} simply",
            "Problems with {}",
            "Future of {}",
            "Benefits of {}",
            "How does {} work?",
            "Top {} tools"
        ]
        
        # Expanded topic categories
        tech_topics = ["machine learning", "python programming", "web development", "cybersecurity", "blockchain", "AI"]
        science_topics = ["climate change", "quantum computing", "space exploration", "renewable energy", "genetics"]
        business_topics = ["cryptocurrency", "startup funding", "remote work", "digital marketing", "e-commerce"]
        general_topics = ["healthy eating", "fitness routines", "travel tips", "cooking recipes", "home improvement"]
        
        all_topics = tech_topics + science_topics + business_topics + general_topics
        
        dataset = []
        failed_attempts = 0
        max_failures = 50
        
        print(f"🔄 Generating {num_examples} training examples...")
        
        for i in tqdm(range(num_examples)):
            if failed_attempts >= max_failures:
                print(f"⚠️  Too many failures ({failed_attempts}), stopping early")
                break
                
            try:
                # Generate query with more variety
                if random.random() < 0.3:  # 30% chance of comparison queries
                    topic1, topic2 = random.sample(all_topics, 2)
                    query = f"Compare {topic1} and {topic2}"
                else:
                    topic = random.choice(all_topics)
                    template = random.choice(query_templates)
                    query = template.format(topic)
                
                # Perform actual search
                results = list(self.ddgs.text(query, max_results=5))
                
                if not results:
                    failed_attempts += 1
                    continue
                
                # Format training example
                example = {
                    "instruction": f"Search the web for: {query}",
                    "input": "",
                    "output": self.format_search_results(query, results)
                }
                
                dataset.append(example)
                failed_attempts = 0  # Reset failure counter
                
                # Rate limiting to avoid being blocked
                time.sleep(random.uniform(0.5, 1.5))
                
            except Exception as e:
                print(f"\n❌ Error generating example {i}: {e}")
                failed_attempts += 1
                time.sleep(2)  # Longer wait on error
                continue
                
        print(f"✅ Generated {len(dataset)} examples successfully!")
        return dataset
    
    def format_search_results(self, query, results):
        """Format search results for training"""
        formatted = f"I searched for: {query}\n\nHere are the results:\n\n"
        
        for i, result in enumerate(results[:3], 1):
            title = result.get('title', 'No title')[:100]
            body = result.get('body', 'No description')[:300]
            href = result.get('href', 'No URL')
            
            formatted += f"**{i}. {title}**\n"
            formatted += f"{body}...\n"
            formatted += f"🔗 Source: {href}\n\n"
            
        return formatted

def main():
    print("🚀 Starting dataset generation...")
    
    generator = WebSearchDatasetGenerator()
    
    # Start with smaller dataset for testing
    training_data = generator.generate_search_examples(500)
    
    # Save dataset
    output_file = 'data/web_search_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved to {output_file}")
    print(f"📊 Total examples: {len(training_data)}")
    
    # Show sample
    if training_data:
        print("\n📝 Sample training example:")
        print("Instruction:", training_data[0]['instruction'])
        print("Output preview:", training_data[0]['output'][:200] + "...")

if __name__ == "__main__":
    main()
```

**Run dataset generation:**
```bash
python generate_dataset.py
```

**What happens:**
- Script will run for 2-3 hours
- You'll see progress bar showing completion
- Creates `data/web_search_dataset.json` with 500 examples
- Each example shows a search query and formatted results

**Expected output:**
```
🚀 Starting dataset generation...
🔄 Generating 500 training examples...
100%|██████████| 500/500 [2:34:12<00:00, 18.45s/it]
✅ Generated 487 examples successfully!
✅ Dataset saved to data/web_search_dataset.json
📊 Total examples: 487
```

---

## Phase 3: Model Training (4-6 hours)

### Step 3.1: Create Training Script
Create file: `train_model.py`
```python
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os
from datetime import datetime
import wandb

# Initialize Weights & Biases for monitoring (optional)
# wandb.init(project="websearch-llm-training")

def load_and_format_dataset(dataset_path):
    """Load and format the dataset for training"""
    print("📂 Loading dataset...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"📊 Loaded {len(raw_data)} examples")
    
    formatted_data = []
    for example in raw_data:
        # Format as Mistral chat template
        prompt = f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
        
        formatted_data.append({
            "text": prompt,
            "instruction": example['instruction'],
            "response": example['output']
        })
    
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer():
    """Setup model and tokenizer with dual RTX 3060 optimization"""
    print("🤖 Loading base model...")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Important for training
    
    # Load model with dual GPU optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,  # 8-bit for better quality with your VRAM
        max_memory={0: "11GB", 1: "11GB"}  # Reserve 1GB per GPU
    )
    
    print(f"✅ Model loaded on devices: {model.device}")
    
    return model, tokenizer

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning"""
    print("🔧 Setting up LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,  # Higher rank for better quality
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset"""
    print("🔤 Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=2048,  # Adjust based on your data
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    print("🚀 Starting training process...")
    print(f"⏰ Start time: {datetime.now()}")
    
    # Load dataset
    dataset = load_and_format_dataset('data/web_search_dataset.json')
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora(model)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Training arguments optimized for dual RTX 3060
    training_args = TrainingArguments(
        output_dir="./models/websearch-model-checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=4,    # Optimized for your setup
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Logging
        logging_dir="./logs",
        report_to=None,  # Set to "wandb" if using W&B
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("🏋️ Starting training...")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    print("💾 Saving model...")
    trainer.save_model("./models/websearch-lora-final")
    tokenizer.save_pretrained("./models/websearch-lora-final")
    
    print(f"✅ Training completed at {datetime.now()}")
    print("🎉 Model saved to ./models/websearch-lora-final")

if __name__ == "__main__":
    main()
```

**Run training:**
```bash
# Make sure you're in the right environment
conda activate websearch-llm

# Start training (this will take 4-6 hours)
python train_model.py
```

**What you'll see during training:**
```
🚀 Starting training process...
⏰ Start time: 2024-01-15 10:30:00
📂 Loading dataset...
📊 Loaded 487 examples
🤖 Loading base model...
✅ Model loaded on devices: cuda:0
🔧 Setting up LoRA...
trainable params: 167,772,160 || all params: 7,407,312,896 || trainable%: 2.2653
🔤 Tokenizing dataset...
🏋️ Starting training...

{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.8765, 'learning_rate': 0.00019, 'epoch': 0.2}
...
```

**Training will:**
- Use both your RTX 3060s automatically
- Show decreasing loss values (good!)
- Save checkpoints every 100 steps
- Take 4-6 hours total
- Use about 20-22GB of your VRAM

---

## Phase 4: Deploy to Ollama (30 minutes)

### Step 4.1: Merge LoRA with Base Model
Create file: `merge_model.py`
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def merge_lora_model():
    """Merge LoRA weights with base model for deployment"""
    print("🔄 Merging LoRA with base model...")
    
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    lora_model_path = "./models/websearch-lora-final"
    output_path = "./models/websearch-merged"
    
    # Load base model
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load and merge LoRA
    print("🔗 Loading and merging LoRA...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print("💾 Saving merged model...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Merged model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    merge_lora_model()
```

**Run merge:**
```bash
python merge_model.py
```

### Step 4.2: Create Ollama Modelfile
Create file: `create_ollama_model.py`
```python
import os

def create_modelfile():
    """Create optimized Modelfile for Ollama"""
    
    modelfile_content = """FROM ./models/websearch-merged

# Performance parameters for dual RTX 3060
PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_batch 512
PARAMETER num_gpu 2
PARAMETER main_gpu 0
PARAMETER split_mode 1
PARAMETER mlock true
PARAMETER numa false

# Stop tokens
PARAMETER stop "<s>"
PARAMETER stop "</s>"
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"

TEMPLATE \"\"\"<s>[INST] {{ .System }}

{{ .Prompt }} [/INST]\"\"\"

SYSTEM \"\"\"You are WebSearchBot, an AI assistant specialized in web search and information retrieval. 

Key behaviors:
- Provide concise, accurate responses
- Focus on the most relevant information first  
- When processing search results, prioritize recent and authoritative sources
- Structure responses with clear headings and bullet points
- Always indicate if information might be outdated

Response format:
1. Direct answer first
2. Supporting details
3. Sources/links if available\"\"\"
"""
    
    with open('WebSearchModelfile', 'w') as f:
        f.write(modelfile_content)
    
    print("✅ Modelfile created!")

def setup_ollama_env():
    """Set up environment variables for Ollama"""
    
    env_vars = {
        'OLLAMA_NUM_PARALLEL': '4',
        'OLLAMA_MAX_LOADED_MODELS': '2', 
        'OLLAMA_GPU_MEMORY': '22GB',
        'OLLAMA_NUM_GPU': '2',
        'OLLAMA_FLASH_ATTENTION': '1',
        'OLLAMA_KEEP_ALIVE': '5m',
        'OLLAMA_MAX_QUEUE': '10',
        'CUDA_VISIBLE_DEVICES': '0,1'
    }
    
    env_script = "#!/bin/bash\n# Ollama optimization script\n\n"
    
    for key, value in env_vars.items():
        env_script += f"export {key}={value}\n"
        os.environ[key] = value
    
    env_script += "\necho 'Environment variables set for Ollama optimization!'\n"
    env_script += "ollama serve &\n"
    env_script += "sleep 5\n"
    env_script += "ollama create websearch-fast -f WebSearchModelfile\n"
    
    with open('setup_ollama.sh', 'w') as f:
        f.write(env_script)
    
    os.chmod('setup_ollama.sh', 0o755)  # Make executable
    
    print("✅ Ollama setup script created!")

if __name__ == "__main__":
    create_modelfile()
    setup_ollama_env()
```

**Run setup:**
```bash
python create_ollama_model.py
```

### Step 4.3: Deploy to Ollama
```bash
# Run the setup script
./setup_ollama.sh

# Or manually:
# 1. Set environment variables and start Ollama
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_GPU_MEMORY=22GB
export OLLAMA_NUM_GPU=2
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE=5m

# 2. Start Ollama server
ollama serve &

# 3. Wait a moment for server to start
sleep 5

# 4. Create the model
ollama create websearch-fast -f WebSearchModelfile

# 5. Test the model
ollama run websearch-fast "What is machine learning?"
```

---

## Phase 5: Test and Optimize (15 minutes)

### Step 5.1: Create Test Script
Create file: `test_model.py`
```python
import ollama
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

def test_basic_functionality():
    """Test basic model functionality"""
    print("🧪 Testing basic functionality...")
    
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Latest developments in quantum computing",
        "Best practices for web development"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        start_time = time.time()
        
        response = ollama.generate(
            model='websearch-fast',
            prompt=f"Search for: {query}",
            options={'temperature': 0.3}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  Response time: {response_time:.2f}s")
        print(f"📝 Response preview: {response['response'][:200]}...")

def test_performance():
    """Benchmark model performance"""
    print("\n🏃 Performance benchmark...")
    
    test_queries = [
        "What is Python programming?",
        "Explain neural networks",
        "Climate change solutions",
        "Blockchain technology basics",
        "Space exploration news"
    ]
    
    times = []
    
    for i, query in enumerate(test_queries):
        start = time.time()
        
        response = ollama.generate(
            model='websearch-fast',
            prompt=f"Quick search: {query}",
            options={
                'temperature': 0.3,
                'num_predict': 256
            }
        )
        
        end = time.time()
        response_time = end - start
        times.append(response_time)
        
        print(f"Query {i+1}: {response_time:.2f}s ({len(response['response'])} chars)")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Average response time: {avg_time:.2f}s")
    print(f"🚀 Throughput: {60/avg_time:.1f} queries/minute")

def test_parallel_requests():
    """Test concurrent request handling"""
    print("\n⚡ Testing parallel requests...")
    
    queries = [
        "Machine learning basics",
        "Web development trends", 
        "Cybersecurity best practices",
        "Data science workflow"
    ]
    
    def make_request(query):
        start = time.time()
        response = ollama.generate(
            model='websearch-fast',
            prompt=f"Search: {query}",
            options={'temperature': 0.3, 'num_predict': 200}
        )
        end = time.time()
        return end - start, len(response['response'])
    
    start_total = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(make_request, queries))
    
    end_total = time.time()
    total_time = end_total - start_total
    
    print(f"🕐 Total time for 4 parallel requests: {total_time:.2f}s")
    print(f"📈 Parallel efficiency: {sum(r[0] for r in results) / total_time:.1f}x")

if __name__ == "__main__":
    print("🚀 Starting model tests...")
    
    try:
        test_basic_functionality()
        test_performance()
        test_parallel_requests()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure Ollama is running and the model is loaded.")
```

**Run tests:**
```bash
python test_model.py
```

**Expected results:**
```
🚀 Starting model tests...
🧪 Testing basic functionality...

🔍 Testing: What is artificial intelligence?
⏱️  Response time: 1.23s
📝 Response preview: I searched for: What is artificial intelligence?

Here are the results:

**1. Artificial Intelligence (AI) - Definition and Overview**
AI refers to the simulation of human intelligence...

📊 Average response time: 1.45s
🚀 Throughput: 41.4 queries/minute
✅ All tests completed successfully!
```

---

## 🔧 Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`
**Solution:**
```bash
# Reduce batch size in train_model.py
per_device_train_batch_size=2  # Instead of 4
gradient_accumulation_steps=4  # Instead of 2
```

### Issue 2: Ollama Model Not Found
**Error:** `model 'websearch-fast' not found`
**Solution:**
```bash
# Check if model exists
ollama list

# Recreate if missing
ollama create websearch-fast -f WebSearchModelfile
```

### Issue 3: Slow Training
**Symptoms:** Very slow training speed
**Solution:**
```bash
# Check GPU utilization
nvidia-smi

# Ensure both GPUs are being used
export CUDA_VISIBLE_DEVICES=0,1
```

### Issue 4: Dataset Generation Fails
**Error:** Search requests being blocked
**Solution:**
```python
# Increase delays in generate_dataset.py
time.sleep(random.uniform(2, 4))  # Longer delays
```

---

## 📈 Monitoring Progress

### During Dataset Generation:
- Watch the progress bar
- Check `data/web_search_dataset.json` file size growing
- Should take 2-3 hours for 500 examples

### During Training:
- Loss should decrease from ~2.5 to ~0.8
- Both GPUs should show 90%+ utilization in `nvidia-smi`
- Training checkpoints saved every 100 steps

### During Testing:
- Response times under 2 seconds for simple queries
- Throughput of 30+ queries/minute
- Parallel requests working without errors

---

## 🎉 Success Indicators

**You've succeeded when:**
- ✅ Dataset generated with 400+ examples
- ✅ Training completes without out-of-memory errors  
- ✅ Model responds to test queries in under 2 seconds
- ✅ Ollama shows both GPUs being utilized
- ✅ Model gives relevant, structured responses about web search topics

**Your final model will:**
- Run entirely locally on your dual RTX 3060s
- Handle web search queries with structured responses
- Support 4+ concurrent users
- Respond in 1-2 seconds for most queries

Ready to start? Just follow each phase in order and you'll have your custom web search LLM running on Ollama! 🚀
