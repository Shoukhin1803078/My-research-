# Take-Home Test: Machine Learning Researcher at BDMLI

Here is the link how I use BERT: https://www.kaggle.com/code/foolofatook/news-classification-using-bert

"""
# Text Summarization with LLAMA2 and News Category Dataset

Here is the code link:https://colab.research.google.com/drive/1xGH2t4m2kTPkCTEnZZDVYaXawMGSPvp-#scrollTo=w_tXfklT3g8D

This project explores the use of LLAMA2, an advanced language model, for the task of text summarization. We focus on the News Category Dataset to create a model capable of generating concise and meaningful summaries of news articles.

## 1. Dataset Selection 

### Option: Text Summarization

#### Dataset: News Category Dataset

For this task, I selected the "News Category Dataset" from Kaggle . The dataset contains over 200,000 news articles categorized into different topics such as politics, entertainment, sports, etc. I choose this dataset because:

- **Data Size**: The dataset is sufficiently large specially it a broad range of news articles (200000 news articales), which is ideal for building and training models or fine-tune Large Language Models (LLMs) effectively.
- **Relevance to Real-World Applications**: Summarizing news efficiently is crucial for content aggregation platforms, enabling readers to quickly access and understand key information. News articles reflect real-world events and topics, making the dataset relevant to natural language processing tasks.
- **Potential for Creative Exploration**: The diverse range of news categories allows for creative exploration using prompt engineering techniques.


## 2. Kaggle Model Identification

###  Model Choosing: LLAMA2

In this scenario here LLAMA2 is better than GEMMA and Falcon 7B because some issues and advantage I mentioned bellow why LLAMA2 is best for this case.Here I explain some reasons for choosing the LLAMA2 model from my evaluation 
(My evaluation Code is avaiable in this link: https://github.com/Shoukhin1803078/My-research-/blob/main/BDMLI.ipynb )
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

Learning from the training operational observation:

  -  Llama2 finetunes faster. This is likely because Llama2–7b is a smaller than gemma-7b.
  -  Llama2 shows better training loss on this finetuning task. Llama2 fits the finetuning data a lot better, but it may also subject to overfitting faster as training epochs increase)
  -  Llama2 outperforms in terms of loading and responding
  -  Llama2 responses a bit faster than Gemma. The response time highly depends on the number of generated tokens. The longer the response, the slower the inference. For my example questions tested on NVIDIA A10G 24G, inference time spans from 0.2s to 40s.

  
- **Capability for Text Analysis Tasks**: The Llama2 model has demonstrated state-of-the-art performance on tasks such as text generation, sentiment analysis, and question answering, aligning with the objectives of the "News Category Dataset".
- **Availability and Accessibility**: The Llama2 model is available on Kaggle and can be easily accessed for experimentation and fine-tuning.

#### Rationale:
- **Relevance**: LLAMA2 is specifically engineered for advanced language understanding and generation, making it highly suitable for producing accurate and contextual summaries.
- **Adaptability**: Given its robust NLP capabilities, LLAMA2 can be fine-tuned to grasp the specific nuances and styles required for effective news summarization.

## 3. Model Capability Evaluation

### LLAMA2

#### Performance Metrics: 
- Utilize metrics like BLEU and ROUGE to measure the quality and accuracy of the generated summaries compared to human-written references.

#### Architecture:
- A comprehensive understanding of LLAMA2’s architecture is critical for optimizing its summarization performance.

#### Training/Fine-tuning Potential:
- Evaluate the feasibility of further training LLAMA2 on the news dataset, taking into account factors like computational resources and the volume of data required for effective learning.
"""


