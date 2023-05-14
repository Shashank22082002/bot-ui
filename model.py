import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
import pandas as pd

# Load the pre-trained BERT model and tokenizer
model_name = "aubmindlab/bert-base-arabertv2"
arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_model = AutoModel.from_pretrained(model_name)

# Preprocess the input text using ArabertPreprocessor
arabert_prep = ArabertPreprocessor(model_name=model_name)

def findEmbedding(str):
    text_preprocessed = arabert_prep.preprocess(str)
    arabert_input = arabert_tokenizer.encode(text_preprocessed, add_special_tokens=True)
    tensor_input_ids = torch.tensor(arabert_input).unsqueeze(0)
    output = arabert_model(tensor_input_ids)
    return output[0][0][1:-1]

def calculateSimilarity(str1, str2):
    emb1 = findEmbedding(str1)
    emb2 = findEmbedding(str2)
    diff = (emb1.shape[0] - emb2.shape[0])
    if (diff > 0):
        emb2 = F.pad(input=emb2, pad=(0, 0, diff, 0), mode='constant', value=0)
    else:
        emb1 = F.pad(input=emb1, pad=(0, 0, -diff, 0), mode='constant', value=0)
    cos = torch.nn.CosineSimilarity(dim=0)
    cos_sim = cos(emb1, emb2)
    overall_similarity = torch.mean(cos_sim)
    return overall_similarity

# Load the dataset
path = "dataset_arabic_temp.xlsx"
df=pd.read_excel(path,usecols='H,J')
df.columns.values.tolist()
QUESTIONS = df.columns.values.tolist()[0]
ANSWER = df.columns.values.tolist()[1]
print(QUESTIONS, ANSWER)

def targetFunction(query):
    max_sim = 0.0
    response = ""
    for ind in df.index:
        question = df[QUESTIONS][ind]
        answer = df[ANSWER][ind]
        print(ind, question);
        sim = calculateSimilarity(query, question)
        if sim > max_sim:
            response = answer
            max_sim = sim
    return {"RESPONSE": response, "CONFIDENCE": max_sim}


# ans = targetFunction("لقد نسيت كلمة السر. ماذا يجب أن أفعل؟");
# english version: ans = targetFunction("What is the grievance handling method?");

# print(ans)