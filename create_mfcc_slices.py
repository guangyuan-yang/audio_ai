import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050
DURATION = 30
# Let’s make sure all files have the same amount of samples, pick a duration right under 30 seconds.
SAMPLES_TRACK = SAMPLE_RATE * (DURATION - 1)


def create_mfcc_slices(filename, folder_output, n_mfcc=13, n_fft=2048, hop_length=512, num_slices=5):
    samples_slice = int(SAMPLES_TRACK / num_slices)
    audio_signal, sample_rate = librosa.load(filename, sr=SAMPLE_RATE)
    for clip in range(num_slices):
        sample_start = samples_slice * clip
        sample_end = sample_start + samples_slice
        mfcc = librosa.feature.mfcc(audio_signal[sample_start:sample_end],
                                    sr=sample_rate,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        plt.figure(figsize=(40/num_slices, 8))
        librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
        if num_slices == 1:
            fig_name = folder_output + os.path.basename(filename).replace('.wav', f"_MFCC_whole.png")
        else:
            fig_name = folder_output + os.path.basename(filename).replace('.wav', f"_MFCC_{clip}.png")
        plt.savefig(fig_name)
        plt.close('all')
