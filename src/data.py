
import os

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

# Основні параметри аудіо
SAMPLE_RATE = 16000
AUDIO_LENGTH = SAMPLE_RATE  # 1 секунда

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
SPEC_WIDTH = 100

CLASSES = ["yes", "no", "up", "down"]


def preprocess_audio(waveform, sample_rate):
    """
    Перетворює аудіосигнал у Mel-Spectrogram
    фіксованого розміру.
    """

    # Перетворення стерео в моно
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Зміна частоти дискретизації
    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Вирівнювання тривалості до 1 секунди
    if waveform.shape[1] < AUDIO_LENGTH:
        waveform = F.pad(
            waveform,
            (0, AUDIO_LENGTH - waveform.shape[1])
        )
    else:
        waveform = waveform[:, :AUDIO_LENGTH]

    # Перетворення в Mel-Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    spectrogram = mel_transform(waveform)

    # Вирівнювання ширини спектрограми
    if spectrogram.shape[2] < SPEC_WIDTH:
        spectrogram = F.pad(
            spectrogram,
            (0, SPEC_WIDTH - spectrogram.shape[2])
        )
    else:
        spectrogram = spectrogram[:, :, :SPEC_WIDTH]

    # Логарифмічне перетворення Mel-Spectrogram
    spectrogram = torch.log(spectrogram.clamp_min(1e-6))

# Стандартизація спектрограми
    mean = spectrogram.mean()
    std = spectrogram.std()

    spectrogram = (spectrogram - mean) / (std + 1e-6)

    return spectrogram


def load_audio_file(file_path):
    """Завантажує WAV-файл і виконує попередню обробку."""

    waveform, sample_rate = torchaudio.load(file_path)

    return preprocess_audio(waveform, sample_rate)


def load_file_list(dataset_dir, classes=CLASSES):
    """Формує список шляхів до аудіофайлів та їхніх класів."""

    files = []

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Директорію класу не знайдено: {class_dir}"
            )

        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(class_dir, filename)
                files.append((file_path, class_name))

    return files


class SpeechCommandsDataset(Dataset):
    """Dataset для навчання CNN на аудіозаписах."""

    def __init__(self, files, classes=CLASSES):
        self.files = files
        self.classes = classes

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(classes)
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path, label = self.files[index]

        spectrogram = load_audio_file(file_path)
        target = self.class_to_idx[label]

        return spectrogram, target
