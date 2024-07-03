import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm

device = 'cuda'
question_f = open("question.txt")
answer_f = open("answer.txt")
question = question_f.readlines()
answer = answer_f.readlines()


class TextSimilarityDataset(Dataset):
    def __init__(self, tokenizer, max_len=128):
        self.data = []
        for i in range(2000):
            text1 = question[i]
            text2 = answer[i]
            similarity_score = 5
            inputs1 = tokenizer(text1, padding='max_length', truncation=True, max_length=max_len)
            inputs2 = tokenizer(text2, padding='max_length', truncation=True, max_length=max_len)
            self.data.append({
                'input_ids1': inputs1['input_ids'],
                'attention_mask1': inputs1['attention_mask'],
                'input_ids2': inputs2['input_ids'],
                'attention_mask2': inputs2['attention_mask'],
                'similarity_score': float(similarity_score),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        embeddings1 = self.dropout(self.bert(input_ids=input_ids1, attention_mask=attention_mask1)['last_hidden_state'])
        embeddings2 = self.dropout(self.bert(input_ids=input_ids2, attention_mask=attention_mask2)['last_hidden_state'])

        embeddings1 = torch.mean(embeddings1, dim=1)
        embeddings2 = torch.mean(embeddings2, dim=1)

        similarity_scores = cosine_similarity_torch(embeddings1, embeddings2)

        normalized_similarities = (similarity_scores + 1) * 2.5
        return normalized_similarities.unsqueeze(1)


class SmoothL1Loss(torch.nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, predictions, targets):
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        quadratic = torch.where(abs_diff < 1, 0.5 * diff ** 2, abs_diff - 0.5)
        return torch.mean(quadratic)


def train_model(model, train_loader,  epochs=50, model_save_path='output/bert_similarity_model.pth'):
    model.to(device)
    criterion = SmoothL1Loss()  # 使用自定义的Smooth L1 Loss
    optimizer = AdamW(model.parameters(), lr=5e-5)  # 调整初始学习率为5e-5
    num_training_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_training_steps,
                                                num_training_steps=num_training_steps)  # 使用带有warmup的余弦退火学习率调度

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            similarity_scores = batch['similarity_score'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(outputs, similarity_scores.unsqueeze(1))
            loss.backward()
            optimizer.step()
            scheduler.step()
    torch.save(model.state_dict(), model_save_path)



def collate_to_tensors(batch):
    input_ids1 = torch.tensor([example['input_ids1'] for example in batch])
    attention_mask1 = torch.tensor([example['attention_mask1'] for example in batch])
    input_ids2 = torch.tensor([example['input_ids2'] for example in batch])
    attention_mask2 = torch.tensor([example['attention_mask2'] for example in batch])
    similarity_score = torch.tensor([example['similarity_score'] for example in batch])

    return {'input_ids1': input_ids1, 'attention_mask1': attention_mask1, 'input_ids2': input_ids2,
            'attention_mask2': attention_mask2, 'similarity_score': similarity_score}


# 加载数据集和预训练模型
tokenizer = BertTokenizer.from_pretrained(r'E:\GEARS_program\VideoRetrieval\bert-base-uncased')
model = BertSimilarityModel(r'E:\GEARS_program\VideoRetrieval\bert-base-uncased')

# 加载数据并创建
train_data = TextSimilarityDataset(tokenizer)
val_data = TextSimilarityDataset(tokenizer)
test_data = TextSimilarityDataset(tokenizer)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_to_tensors)

optimizer = AdamW(model.parameters(), lr=2e-5)

# 开始训练
train_model(model, train_loader)

