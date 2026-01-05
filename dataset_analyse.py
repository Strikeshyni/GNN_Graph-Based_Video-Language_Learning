import json
import numpy as np
from collections import Counter
from src.data.dataset import AVQA_Dataset

# 1. Analyser la distribution des réponses
with open('data/MUSIC-AVQA/train_annotations.json') as f:
    train = json.load(f)
with open('data/MUSIC-AVQA/val_annotations.json') as f:
    val = json.load(f)

# Distribution des réponses
train_answers = [d.get('anser', d.get('answer', '')) for d in train]
val_answers = [d.get('anser', d.get('answer', '')) for d in val]

train_counter = Counter(train_answers)
val_counter = Counter(val_answers)

print('='*60)
print('1. DISTRIBUTION DES RÉPONSES')
print('='*60)
print(f'\nNombre de classes: {len(train_counter)}')
print(f'Train samples: {len(train)}')
print(f'Val samples: {len(val)}')

print('\nTop 15 réponses (train):')
for ans, count in train_counter.most_common(15):
    pct = 100 * count / len(train)
    print(f'  {ans}: {count} ({pct:.1f}%)')

# Baseline (toujours prédire la réponse la plus fréquente)
most_common = train_counter.most_common(1)[0]
baseline_acc = 100 * most_common[1] / len(train)
print(f'\nBaseline accuracy (always predict \"{most_common[0]}\"): {baseline_acc:.1f}%')

# 2. Overlap entre train et val
print('\n' + '='*60)
print('2. OVERLAP TRAIN/VAL')
print('='*60)
train_answers_set = set(train_answers)
val_answers_set = set(val_answers)
overlap = train_answers_set & val_answers_set
print(f'Réponses dans train: {len(train_answers_set)}')
print(f'Réponses dans val: {len(val_answers_set)}')
print(f'Overlap: {len(overlap)} ({100*len(overlap)/len(val_answers_set):.1f}%)')

# Réponses dans val mais pas dans train
val_only = val_answers_set - train_answers_set
if val_only:
    print(f'\n⚠️ Réponses dans VAL mais PAS dans TRAIN: {val_only}')

print('\n' + '='*60)
print('3. ANALYSE DES TYPES DE QUESTIONS')
print('='*60)

with open('data/MUSIC-AVQA/train_annotations.json') as f:
    train = json.load(f)

# Types de questions
import ast
type_counter = Counter()
for d in train:
    try:
        q_type = ast.literal_eval(d.get('type', '[]'))
        if q_type:
            type_counter[tuple(q_type)] += 1
    except:
        pass

print('\nDistribution des types de questions:')
for q_type, count in type_counter.most_common():
    pct = 100 * count / len(train)
    print(f'  {q_type}: {count} ({pct:.1f}%)')

# Réponses par type
print('\n' + '='*60)
print('4. RÉPONSES PAR TYPE DE QUESTION')
print('='*60)

type_answers = {}
for d in train:
    try:
        q_type = ast.literal_eval(d.get('type', '[]'))
        if q_type:
            key = tuple(q_type)
            if key not in type_answers:
                type_answers[key] = []
            type_answers[key].append(d.get('anser', d.get('answer', '')))
    except:
        pass

for q_type in list(type_counter.keys())[:6]:
    answers = type_answers.get(q_type, [])
    counter = Counter(answers)
    print(f'\n{q_type}:')
    for ans, count in counter.most_common(5):
        pct = 100 * count / len(answers)
        print(f'    {ans}: {count} ({pct:.1f}%)')

print('='*60)
print('5. VÉRIFICATION DES FEATURES')
print('='*60)

dataset = AVQA_Dataset('./data/MUSIC-AVQA', split='train')

# Vérifier plusieurs samples
real_features = 0
random_features = 0

for i in range(min(50, len(dataset))):
    sample = dataset[i]
    audio = sample['audio_feat']
    visual = sample['visual_feat']
    
    # Les features réelles ont généralement des valeurs entre 0 et quelques unités
    # Les features aléatoires (torch.randn) ont mean~0 et std~1
    
    audio_std = audio.std().item()
    visual_std = visual.std().item()
    
    # Les features VGGish sont généralement plus petites (0.2-0.5 std)
    # Les features CLIP sont autour de 0.5-2 std
    if audio_std < 0.8 and visual_std < 3:
        real_features += 1
    else:
        random_features += 1

print(f'\nSur 50 samples:')
print(f'  Features réelles: {real_features}')
print(f'  Features aléatoires: {random_features}')

# Analyser un sample en détail
print('\n' + '='*60)
print('6. EXEMPLE DE SAMPLE')
print('='*60)
sample = dataset[0]
print(f"\nVideo ID: {sample['video_id']}")
print(f"Question: {dataset.annotations[0]['question']}")
print(f"Answer: {dataset.annotations[0]['answer']}")
print(f"\nAudio shape: {sample['audio_feat'].shape}")
print(f"Audio stats: mean={sample['audio_feat'].mean():.4f}, std={sample['audio_feat'].std():.4f}")
print(f"Visual shape: {sample['visual_feat'].shape}")
print(f"Visual stats: mean={sample['visual_feat'].mean():.4f}, std={sample['visual_feat'].std():.4f}")
print(f"Question feat shape: {sample['question_feat'].shape}")
print(f"Question feat stats: mean={sample['question_feat'].mean():.4f}, std={sample['question_feat'].std():.4f}")

import torch
import torch.nn.functional as F
from collections import Counter
from src.data.dataset import AVQA_Dataset
from src.models.avqa_gnn import AVQA_GNN
import argparse

print('='*60)
print('7. ANALYSE DES PRÉDICTIONS DU MODÈLE')
print('='*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = AVQA_Dataset('./data/MUSIC-AVQA', split='val')

args = argparse.Namespace(
    num_classes=42, node_dim=512, edge_dim=512, out_channels=512,
    gnn_layers=2, gnn_heads=4, dropout=0.1, gnn_type='gat'
)

model = AVQA_GNN(args).to(device)
checkpoint = torch.load('checkpoints/avqa_gnn_gat_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

predictions = []
ground_truths = []
confidences = []

with torch.no_grad():
    for i in range(min(200, len(dataset))):
        sample = dataset[i]
        
        audio = sample['audio_feat'].unsqueeze(0).to(device)
        visual = sample['visual_feat'].unsqueeze(0).to(device)
        question = sample['question_feat'].unsqueeze(0).to(device)
        
        sg = sample['sg_data']
        qg = sample['qg_data']
        
        sg_edge_attr = getattr(sg, 'edge_attr', torch.zeros(sg.edge_index.shape[1], 512))
        qg_edge_attr = getattr(qg, 'edge_attr', torch.zeros(qg.edge_index.shape[1], 512))
        
        sg_batch = torch.zeros(sg.x.shape[0], dtype=torch.long)
        qg_batch = torch.zeros(qg.x.shape[0], dtype=torch.long)
        
        sg_data = (sg.x.to(device), sg.edge_index.to(device), sg_edge_attr.to(device), sg_batch.to(device))
        qg_data = (qg.x.to(device), qg.edge_index.to(device), qg_edge_attr.to(device), qg_batch.to(device))
        
        output = model(audio, visual, question, sg_data, qg_data)
        probs = F.softmax(output, dim=-1)
        conf, pred = probs.max(dim=-1)
        
        predictions.append(pred.item())
        ground_truths.append(sample['answer'].item())
        confidences.append(conf.item())

# Statistiques
pred_counter = Counter(predictions)
gt_counter = Counter(ground_truths)

idx2answer = dataset.idx_to_answer

print(f'\nDistribution des prédictions (Top 10):')
for idx, count in pred_counter.most_common(10):
    pct = 100 * count / len(predictions)
    print(f'  {idx2answer[idx]:15s}: {count:3d} ({pct:5.1f}%)')

print(f'\nDistribution des ground truths (Top 10):')
for idx, count in gt_counter.most_common(10):
    pct = 100 * count / len(ground_truths)
    print(f'  {idx2answer[idx]:15s}: {count:3d} ({pct:5.1f}%)')

print(f'\nConfidence moyenne: {sum(confidences)/len(confidences):.3f}')
print(f'Confidence min/max: {min(confidences):.3f} / {max(confidences):.3f}')

unique_preds = len(set(predictions))
print(f'\nNombre de classes différentes prédites: {unique_preds}/42')

correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
print(f'Accuracy sur ces 200 samples: {100*correct/len(predictions):.1f}%')

print('='*60)
print('8. ANALYSE DU COLLAPSE')
print('='*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = AVQA_Dataset('./data/MUSIC-AVQA', split='val')

args = argparse.Namespace(
    num_classes=42, node_dim=512, edge_dim=512, out_channels=512,
    gnn_layers=2, gnn_heads=4, dropout=0.1, gnn_type='gat'
)

model = AVQA_GNN(args).to(device)
checkpoint = torch.load('checkpoints/avqa_gnn_gat_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prendre un sample
sample = dataset[0]

audio = sample['audio_feat'].unsqueeze(0).to(device)
visual = sample['visual_feat'].unsqueeze(0).to(device)
question = sample['question_feat'].unsqueeze(0).to(device)

sg = sample['sg_data']
qg = sample['qg_data']

sg_edge_attr = getattr(sg, 'edge_attr', torch.zeros(sg.edge_index.shape[1], 512))
qg_edge_attr = getattr(qg, 'edge_attr', torch.zeros(qg.edge_index.shape[1], 512))

sg_batch = torch.zeros(sg.x.shape[0], dtype=torch.long)
qg_batch = torch.zeros(qg.x.shape[0], dtype=torch.long)

sg_data = (sg.x.to(device), sg.edge_index.to(device), sg_edge_attr.to(device), sg_batch.to(device))
qg_data = (qg.x.to(device), qg.edge_index.to(device), qg_edge_attr.to(device), qg_batch.to(device))

with torch.no_grad():
    output = model(audio, visual, question, sg_data, qg_data)
    probs = F.softmax(output, dim=-1)

# Analyser les logits
print(f'\nRaw logits (sample 0):')
print(f'  Min: {output.min().item():.3f}')
print(f'  Max: {output.max().item():.3f}')
print(f'  Range: {(output.max() - output.min()).item():.3f}')

idx2answer = dataset.idx_to_answer
n_classes = len(idx2answer)

print(f'\nNombre de classes dans le modèle: {model.classifier[-1].out_features}')
print(f'Nombre de classes dans idx2answer: {n_classes}')

print(f'\nTop 10 classes par logit:')
top10 = output[0].topk(min(10, n_classes))
for i, (val, idx) in enumerate(zip(top10.values, top10.indices)):
    if idx.item() < n_classes:
        print(f'  {i+1}. {idx2answer[idx.item()]:15s}: logit={val.item():.3f}, prob={probs[0,idx].item():.3f}')
    else:
        print(f'  {i+1}. idx={idx.item()} (>n_classes): logit={val.item():.3f}')

# Vérifier le biais
print(f'\n' + '='*60)
print('9. PROBLÈME DÉTECTÉ: ZERO est toujours favorisé')
print('='*60)
zero_idx = dataset.answer_to_idx['zero']
yes_idx = dataset.answer_to_idx['yes']
no_idx = dataset.answer_to_idx['no']
two_idx = dataset.answer_to_idx['two']

print(f"\nLogit pour 'zero' (idx={zero_idx}): {output[0, zero_idx].item():.3f}")
print(f"Logit pour 'yes' (idx={yes_idx}): {output[0, yes_idx].item():.3f}")
print(f"Logit pour 'no' (idx={no_idx}): {output[0, no_idx].item():.3f}")
print(f"Logit pour 'two' (idx={two_idx}): {output[0, two_idx].item():.3f}")

# Calculer la différence avec 2ème classe
sorted_logits = output[0].sort(descending=True)
print(f"\nDifférence entre 1ère et 2ème classe: {(sorted_logits.values[0] - sorted_logits.values[1]).item():.3f}")
print(f"C'est faible, donc le modèle n'est pas très confiant")