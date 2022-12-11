import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as lplt
seed = 12
np.random.seed(seed)

audio_fp = './dataset/train/blues.00000.wav'
audio_data, sr = librosa.load(audio_fp)
audio_data, _ = librosa.effects.trim(audio_data)

# plot sample file
plt.figure(figsize=(15,5))
lplt.waveplot(audio_data)
plt.show()

# Default FFT window size
n_fft = 2048 # window size
hop_length = 512 # window hop length for STFT

stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
stft_db = librosa.amplitude_to_db(stft, ref=np.max)

plt.figure(figsize=(12,4))
lplt.specshow(stft, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Spectrogram with amplitude")
plt.show()

plt.figure(figsize=(12,4))
lplt.specshow(stft_db, sr=sr, x_axis='time', y_axis='log', cmap='cool')
plt.colorbar()
plt.title("Spectrogram with decibel log")
plt.show()

# plot zoomed audio wave 
start = 1000
end = 1200
plt.figure(figsize=(16,4))
plt.plot(audio_data[start:end])
plt.show()

# Mel-spectrogram
mel_spec = librosa.feature.melspectrogram(audio_data, sr=sr)
mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
plt.figure(figsize=(16,6))
lplt.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')
plt.colorbar()
plt.title("Mel Spectrogram")
plt.show()

# Chroma Features
chroma = librosa.feature.chroma_stft(audio_data, sr=sr)
plt.figure(figsize=(16,6))
lplt.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
plt.title("Chroma Features")
plt.show()

# fnames = ['blues.00000.wav', 'classical.00000.wav','country.00000.wav','disco.00000.wav','hiphop.00000.wav'
#           'jazz.00000.wav','metal.00000.wav','pop.00000.wav','reggae.00000.wav','rock.00000.wav']
# audio_fp = './dataset/train/blues.00000.wav'
# fig, axs = plt.subplots(5, 2)

# for i in range(5):
#     for j in range(2):
#         fname = fnames[i*2 + j]
#         audio_fp = os.path.join('./dataset/train', fname)
#         audio_data, sr = librosa.load(audio_fp)
#         audio_data, _ = librosa.effects.trim(audio_data)
        
#         mel_spec = librosa.feature.melspectrogram(audio_data, sr=sr)
#         mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
#         axs[0, 0].lplt.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')
#         axs[0, 0].set_title(fname)
# plt.show()