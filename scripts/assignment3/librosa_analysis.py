import librosa
import librosa.display
import matplotlib.pyplot as plt

ground_truth = "recordings/ground_truth/test.wav"

system = "recordings/google_maps/test.wav"

y1,sr1 = librosa.load(
    ground_truth,
    sr=None
)

y2,sr2 = librosa.load(
    system,
    sr=None
)

plt.figure(figsize=(12,4))

plt.subplot(2,1,1)

librosa.display.waveshow(
    y1,
    sr=sr1
)

plt.title("Ground Truth")

plt.subplot(2,1,2)

librosa.display.waveshow(
    y2,
    sr=sr2
)

plt.title("System Output")

plt.tight_layout()

plt.show()