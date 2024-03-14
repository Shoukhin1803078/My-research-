"""
# Home Test Answer: 

## 1. Dataset Selection : Text Summarization

### Choosing Dataset: News Category Dataset

For this task, I selected the "News Category Dataset" from Kaggle . The dataset contains over 200,000 news articles categorized into different topics such as politics, entertainment, sports, etc. I choose this dataset because:

- **Data Size**: The dataset is sufficiently large specially it a broad range of news articles (200000 news articales), which is ideal for building and training models or fine-tune Large Language Models (LLMs) effectively.
- **Relevance to Real-World Applications**: Summarizing news efficiently is crucial for content aggregation platforms, enabling readers to quickly access and understand key information. News articles reflect real-world events and topics, making the dataset relevant to natural language processing tasks.
- **Potential for Creative Exploration**: The diverse range of news categories allows for creative exploration using prompt engineering techniques.


## 2. Kaggle Model Identification

###  Model Choosing: LLAMA2

In this scenario here LLAMA2 is better than GEMMA and Falcon 7B because some issues and advantage I mentioned bellow why LLAMA2 is best fit for this case. Here I explain some reasons for choosing the LLAMA2 model from my evaluation
  -  Llama2 finetunes faster. 
  -  Llama2 shows better training loss on this finetuning task. Llama2 fits the finetuning data a lot better, but it may also subject to overfitting faster as training epochs increase)
  -  Llama2 outperforms in terms of loading and responding
  -  Llama2 responses a bit faster than Gemma. The response time highly depends on the number of generated tokens. The longer the response, the slower the inference. For my example questions tested on NVIDIA A10G 24G, inference time spans from 0.2s to 40s.
  -  Llama2–7b is a smaller than gemma-7b so it is faster to fintune .

#### Limitation:  
 This model has some disadvantage too. Here I mentioned, 
   - **Dependence on Training Data:** The performance of LLaMA-2 is heavily dependent on the quality and diversity of its training data. If there are gaps or limitations in this data, the model's performance can be negatively affected in those areas.
   - **Adaptability and Learning Over Time:** Unlike humans, LLaMA-2 cannot learn or adapt from new information in real-time once it has been trained. This means it may become outdated or less relevant as language, culture, and world knowledge evolve.

Though it has some limitation but this model performs better than other model 



## 3. Model Capability Evaluation

Here I show some reasons for choosing the LLAMA2 model from my evaluation

(My evaluation Code is given in this link: https://github.com/Shoukhin1803078/My-research-/blob/main/BDMLI.ipynb )


#### Finetuning Setup
    Machines(single GPU): NVIDIA A10G 24G 
    CUDA Version: 12.2
  
#### Base models:
    Gemma model: gemma-7b-it
    Llama2 model: llama-2–7b-chat
  
#### A few SFTTrainer Configuration(same for Gemma and Llama):
    batch_size: 4
    max_steps: 300
    packing: true
    PEFT method: LoRA

### Fine Tuning result:
I have gathered a set of operational metrics. 

![Screenshot 2024-03-13 200614](https://github.com/Shoukhin1803078/My-research-/assets/62458402/633e98c1-050b-4798-b4b2-75b8455bb0d4)

### Observation:

From my evaluation code and result (https://github.com/Shoukhin1803078/My-research-/blob/main/BDMLI.ipynb) we can say that,

  -  As Llama2–7b is a smaller than gemma-7b so it is faster to fintune .
  -  Llama2 shows better training loss on this finetuning task. Llama2 fits the finetuning data a lot better, but it may also subject to overfitting faster as training epochs increase)
  -  Llama2 outperforms in terms of loading and responding
  -  Llama2 responses a bit faster than Gemma. The response time highly depends on the number of generated tokens. The longer the response, the slower the inference. For my example questions tested on NVIDIA A10G 24G, inference time spans from 0.2s to 40s.
  
- **Capability for Text Analysis Tasks**: The Llama2 model has demonstrated state-of-the-art performance on tasks such as text generation, sentiment analysis, and question answering, aligning with the objectives of the "News Category Dataset".
- **Availability and Accessibility**: The Llama2 model is available on Kaggle and can be easily accessed for experimentation and fine-tuning.
- **Model Architecture and Training :** LLaMA-2 have an architecture or training approach that is particularly well-suited for understanding and condensing text. This could involve more effective ways of capturing the essence of a text and reproducing it succinctly.

- **Training Data :** The datasets used for training LLaMA-2 could include a wide variety of texts that make it more proficient in summarization,text catagory tasks. If these datasets are more diverse or comprehensive in the context of summarization, this could give LLaMA-2 an edge.

- **Benchmark Performance :** It's possible that in benchmarks or comparative studies, LLaMA-2 has demonstrated superior performance in text summarization tasks. These benchmarks would typically measure accuracy, coherence, conciseness, and the ability to capture key information.

- **Fine-Tuning and Customization:** LLaMA-2 may offer better options for fine-tuning or customization for specific types of summarization tasks, which can be crucial for achieving high-quality results in diverse applications.

#### Rationale:
- **Relevance**: LLAMA2 is specifically engineered for advanced language understanding and generation, making it highly suitable for producing accurate and contextual summaries.
- **Adaptability**: Given its robust NLP capabilities, LLAMA2 can be fine-tuned to grasp the specific nuances and styles required for effective news summarization.

# Code : https://github.com/Shoukhin1803078/My-research-/blob/main/BDMLI.ipynb

"""


