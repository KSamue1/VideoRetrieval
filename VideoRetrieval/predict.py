import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def cosine_similarity_torch(vec1, vec2, eps=1e-8):
    dot_product = torch.mm(vec1, vec2.t())
    norm1 = torch.norm(vec1, 2, dim=1, keepdim=True)
    norm2 = torch.norm(vec2, 2, dim=1, keepdim=True)
    similarity_scores = dot_product / (norm1 * norm2.t()).clamp(min=eps)
    return similarity_scores


class BertSimilarityModel(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(BertSimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = torch.nn.Dropout(p=0.1)  # 引入Dropout层以防止过拟合

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        pass



tokenizer = BertTokenizer.from_pretrained(r'E:\GEARS_program\VideoRetrieval\bert-base-uncased')
model = BertSimilarityModel(r'E:\GEARS_program\VideoRetrieval\bert-base-uncased')
model.load_state_dict(torch.load(r'E:\GEARS_program\VideoRetrieval\output\bert_similarity_model.pth'))  # 请确保路径正确
model.eval()


def calculate_similarity(text1, text2):

    inputs1 = tokenizer(text1, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    inputs2 = tokenizer(text2, padding='max_length', truncation=True, max_length=128, return_tensors='pt')


    with torch.no_grad():
        embeddings1 = model.bert(**inputs1.to('cpu'))['last_hidden_state'][:, 0]
        embeddings2 = model.bert(**inputs2.to('cpu'))['last_hidden_state'][:, 0]
        similarity_score = cosine_similarity_torch(embeddings1, embeddings2).item()

    normalized_similarity = (similarity_score + 1) * 2.5

    return normalized_similarity

f = open(r"E:\GEARS_program\VideoRetrieval\video_id.txt")
video_info = list(set(f.readlines()))
dict = {}
for i in range(len(video_info)):
    text = video_info[i].split('\t')
    video_info[i] = text[0]
    dict[text[0]] = text[1].replace('\n','')

while(True):
    best_sim = 0
    question = input("Question:")
    for info in video_info:
        similarity = calculate_similarity(question,info)
        if (similarity > best_sim):
            best_sim = similarity
            best_info = info
    print(f"Video title:{best_info} Video id:{dict[best_info]}")