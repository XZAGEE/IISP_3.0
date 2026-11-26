
import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import CLASSES, SpeechCommandsDataset, load_file_list
from src.model import AudioCNN


def set_seed(seed):
    """Фіксує генератори випадкових чисел для відтворюваності."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)

    torch.set_num_threads(2)


def train_model(model, train_loader, device, epochs):
    """Навчає модель протягом заданої кількості епох."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Епоха [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.4f}",
            flush=True
        )


def evaluate_model(model, test_loader, device):
    """Обчислює точність моделі на тестовій вибірці."""

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Навчання CNN для класифікації аудіо"
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=CLASSES,
        help="Класи для навчання"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Кількість епох"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Випадкове зерно"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="artifacts",
        help="Директорія для збереження моделі"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ci_dataset",
        help="Директорія з навчальним датасетом"
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs має бути не менше 1")

    if len(args.classes) < 2:
        parser.error("Потрібно щонайменше два класи")

    if len(set(args.classes)) != len(args.classes):
        parser.error("Назви класів не повинні повторюватися")

    unknown_classes = set(args.classes) - set(CLASSES)

    if unknown_classes:
        parser.error(f"Невідомі класи: {sorted(unknown_classes)}")

    set_seed(args.seed)

    device = torch.device("cpu")

    print("Пристрій:", device, flush=True)
    print("Класи:", args.classes, flush=True)
    print("Кількість епох:", args.epochs, flush=True)

    # Завантаження списку аудіофайлів
    files = load_file_list(args.dataset, args.classes)

    print("Знайдено аудіофайлів:", len(files), flush=True)

    # Розподіл на навчальну та тестову вибірки
    train_files, test_files = train_test_split(
        files,
        test_size=0.2,
        random_state=args.seed,
        stratify=[label for _, label in files]
    )

    print("Навчальна вибірка:", len(train_files), flush=True)
    print("Тестова вибірка:", len(test_files), flush=True)

    # Створення датасетів
    train_dataset = SpeechCommandsDataset(
        train_files,
        classes=args.classes
    )

    test_dataset = SpeechCommandsDataset(
        test_files,
        classes=args.classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # Створення та навчання моделі
    model = AudioCNN(num_classes=len(args.classes)).to(device)

    print("\nПочаток навчання...", flush=True)

    train_model(
        model,
        train_loader,
        device,
        args.epochs
    )

    # Оцінювання
    print("\nТестування моделі...", flush=True)

    accuracy = evaluate_model(
        model,
        test_loader,
        device
    )

    print(
        f"Точність на тестовій вибірці: {accuracy * 100:.2f}%",
        flush=True
    )

    # Збереження результатів
    os.makedirs(args.out, exist_ok=True)

    model_path = os.path.join(args.out, "model.pt")
    labels_path = os.path.join(args.out, "labels.json")
    metrics_path = os.path.join(args.out, "metrics.json")

    torch.save(model.state_dict(), model_path)

    with open(labels_path, "w", encoding="utf-8") as file:
        json.dump(args.classes, file, ensure_ascii=False, indent=4)

    metrics = {
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "model_size_kb": round(
            os.path.getsize(model_path) / 1024, 2
        ),
        "epochs": args.epochs,
        "seed": args.seed,
        "classes": args.classes,
        "train_samples": len(train_files),
        "test_samples": len(test_files)
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    print("\nРезультати збережено:", flush=True)
    print(model_path, flush=True)
    print(labels_path, flush=True)
    print(metrics_path, flush=True)


if __name__ == "__main__":
    main()
