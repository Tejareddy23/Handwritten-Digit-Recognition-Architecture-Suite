"""
NLP Text Dataset, Tokenization, and Vocabulary Construction.

DL Concepts:
- Tokenization: Splitting raw natural text into discrete linguistic units (tokens).
- Vocabulary Building: Indexing words into integer tokens [0, |V|-1] with <PAD> and <UNK>.
- Sequence Padding: Aligning varying sentence lengths to fixed tensor shapes for mini-batching.
"""

import re
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import DATA_DIR, RESULTS_DIR


CATEGORIES = ["Technology", "Finance", "Sports", "Support"]

SEED_TEXT_DATA = [
    # Technology
    ("Python release brings faster execution and improved deep learning libraries", 0),
    ("The neural network architecture achieved state of the art accuracy on benchmarks", 0),
    ("Developers debugged memory leaks in the distributed backend microservices", 0),
    ("Artificial intelligence and transformer models are advancing computer vision", 0),
    ("Deploying containerized docker images on kubernetes cloud clusters", 0),
    ("The open source github repository received thousands of developer stars", 0),
    ("Training convolutional layers with backpropagation and adam optimizer", 0),
    ("Hardware accelerators like GPU and TPU speed up tensor matrix computations", 0),
    ("New firmware update fixes critical security vulnerability in the operating system", 0),
    ("Database indexing optimized sql query latency across relational tables", 0),
    ("Web application frontend built with react typescript and interactive ui", 0),
    ("Machine learning pipeline includes data preprocessing feature extraction and evaluation", 0),

    # Finance
    ("Stock market indexes rallied following positive central bank interest rate cuts", 1),
    ("The quarterly earnings report showed double digit revenue and profit growth", 1),
    ("Investors diversified their portfolio into treasury bonds and commodities", 1),
    ("Venture capital firms invested millions in seed stage fintech startups", 1),
    ("Inflation rates cooled down according to the latest consumer price index", 1),
    ("Cryptocurrency market experienced high volatility during overnight trading", 1),
    ("Corporate balance sheet reflects strong cash reserves and reduced debt liability", 1),
    ("Dividend payout announced for registered shareholders of the public company", 1),
    ("Commercial banks tightened mortgage lending standards amidst economic uncertainty", 1),
    ("Merger and acquisition deal valued at five billion dollars approved by regulators", 1),
    ("Foreign exchange currency exchange rates shifted against the global dollar", 1),
    ("Wall street analysts upgraded the equity price target following fiscal guidance", 1),

    # Sports
    ("The football striker scored a hat trick to win the championship final", 2),
    ("Olympic athlete broke the world record in the hundred meter sprint", 2),
    ("Basketball playoffs game went into overtime with a buzzer beater shot", 2),
    ("Tennis champion advanced to the grand slam finals in straight sets", 2),
    ("The coach substituted tactical players during the second half of the match", 2),
    ("Marathon runners completed twenty six miles through challenging mountain terrain", 2),
    ("Soccer goalkeeper made spectacular diving saves to secure clean sheet", 2),
    ("Formula one driver clinched pole position with the fastest qualifying lap", 2),
    ("Swimming relay team won gold medal setting a new national record", 2),
    ("Baseball pitcher struck out twelve batters in a shutout victory", 2),
    ("Athletes undergoing intense conditioning training ahead of the tournament", 2),
    ("The home stadium crowd erupted in cheers as the trophy was hoisted", 2),

    # Support / Customer Feedback
    ("I need help with my recent order delivery that has not arrived yet", 3),
    ("The customer support representative resolved my account login issue quickly", 3),
    ("Can I please request a full refund because the delivered item was damaged", 3),
    ("Thank you very much for the excellent and courteous technical assistance", 3),
    ("My billing subscription was charged twice and I need an invoice reversal", 3),
    ("The mobile application keeps crashing when I open the settings menu", 3),
    ("How do I update my shipping address before the package is dispatched", 3),
    ("Extremely dissatisfied with the delayed response from the warranty department", 3),
    ("Please cancel my membership renewal and remove my saved payment method", 3),
    ("I received the wrong package and would like an exchange replacement", 3),
    ("Help desk ticket was closed without addressing my password reset inquiry", 3),
    ("Outstanding product quality and very responsive customer service experience", 3)
]


class Vocabulary:
    """
    Bidirectional mapping between words and token IDs.
    """

    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}

    def build_from_texts(self, texts, min_freq=1):
        word_counts = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        idx = len(self.word2idx)
        for token, count in word_counts.items():
            if count >= min_freq and token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

    @staticmethod
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    def text_to_indices(self, text, max_len=25):
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens]
        if len(indices) < max_len:
            indices += [self.word2idx[self.pad_token]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

    def save(self, file_path):
        with open(file_path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f, indent=2)

    @classmethod
    def load(cls, file_path):
        vocab = cls()
        with open(file_path, "r") as f:
            data = json.load(f)
            vocab.word2idx = data["word2idx"]
            vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab

    def __len__(self):
        return len(self.word2idx)


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset yielding tokenized index tensors and category labels.
    """

    def __init__(self, texts, labels, vocab, max_len=25):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.vocab.text_to_indices(self.texts[idx], max_len=self.max_len)
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def generate_augmented_text_data():
    """
    Generates an augmented dataset of text samples across all 4 categories.
    """
    texts = []
    labels = []

    # Replicate and perturb seed dataset to simulate varied text distribution
    variations = [
        "", "please note that ", "latest report confirms ", "breaking update: ",
        "in recent analysis ", "according to experts ", "users reported that ",
        "official announcement says "
    ]

    for prefix in variations:
        for text, label in SEED_TEXT_DATA:
            augmented = (prefix + text).strip()
            texts.append(augmented)
            labels.append(label)

    return texts, labels


def get_nlp_dataloaders(batch_size=16, max_len=25):
    """
    Builds vocabulary and creates Train/Test DataLoaders for text classification.
    """
    from sklearn.model_selection import train_test_split

    texts, labels = generate_augmented_text_data()

    # Stratified split to ensure balanced representation across all categories
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    vocab = Vocabulary()
    vocab.build_from_texts(train_texts)

    train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, max_len=max_len)
    test_dataset = TextClassificationDataset(test_texts, test_labels, vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab
