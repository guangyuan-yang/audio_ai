import os
import sys
import numpy as np
from scipy import stats
import librosa
import librosa.display as lib_disp


class AudioFeatures:
    @staticmethod
    def calculate_feature_stats(feature_name, feature_value):
        feature_stats = dict()
        feature_stats[f'{feature_name}_mean'] = np.mean(feature_value, axis=1).item()
        feature_stats[f'{feature_name}_std'] = np.std(feature_value, axis=1).item()
        feature_stats[f'{feature_name}_skew'] = stats.skew(feature_value, axis=1).item()
        feature_stats[f'{feature_name}_kurtosis'] = stats.kurtosis(feature_value, axis=1).item()
        feature_stats[f'{feature_name}_median'] = np.median(feature_value, axis=1).item()
        feature_stats[f'{feature_name}_min'] = np.min(feature_value, axis=1).item()
        feature_stats[f'{feature_name}_max'] = np.max(feature_value, axis=1).item()

        return feature_stats

    def __init__(self, filename=None, sample_rate=22050, duration=29):
        self.filename = os.path.abspath(os.path.expanduser(filename))
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = sample_rate * duration

    def convert_mp3_to_wav(self):
        import audioread
        import wave
        import contextlib

        if not os.path.exists(self.filename):
            print("File not found.", file=sys.stderr)
            sys.exit(1)
        try:
            with audioread.audio_open(self.filename) as f:
                print(f"Input file: {f.channels} channels at {f.samplerate} Hz; {f.duration} seconds.", file=sys.stderr)
                print(f"Backend: {str(type(f).__module__).split('.')[1]}", file=sys.stderr)
                wav_file = self.filename.replace(".mp3", ".wav")
                with contextlib.closing(wave.open(wav_file, "w")) as of:
                    of.setnchannels(f.channels)
                    of.setframerate(f.samplerate)
                    of.setsampwidth(2)
                    for buf in f:
                        of.writeframes(buf)
            print("File is converted to wav file.")
        except audioread.DecodeError:
            print("File could not be converted.", file=sys.stderr)
            sys.exit(1)
        return wav_file

    def load_wav_file(self):
        if ".wav" in self.filename:
            wav_file = self.filename
        elif ".mp3" in self.filename:
            wav_file = self.convert_mp3_to_wav()
        else:
            raise Exception("File could not be loaded.")

        audio_signal, sample_rate = librosa.load(wav_file, sr=self.sample_rate)
        return audio_signal, sample_rate

    def extract_rsme(self, audio_signal=None, frame_length=2048, hop_length=512, sample_start=0, sample_end=None):
        if sample_end is None:
            sample_end = self.num_samples

        rmse = librosa.feature.rms(y=audio_signal[sample_start:sample_end], frame_length=frame_length,
                                   hop_length=hop_length)
        return rmse

    def extract_zero_crossing_rate(self, audio_signal=None, frame_length=2048, hop_length=512, sample_start=0,
                                   sample_end=None):
        if sample_end is None:
            sample_end = self.num_samples
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_signal[sample_start:sample_end],
                                                                frame_length=frame_length, hop_length=hop_length)
        return zero_crossing_rate

    def extract_spectral_centroid(self, audio_signal=None, sample_rate=None, n_fft=2048, hop_length=512, sample_start=0,
                                  sample_end=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if sample_end is None:
            sample_end = self.num_samples
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_signal[sample_start:sample_end], sr=sample_rate,
                                                              n_fft=n_fft, hop_length=hop_length)
        return spectral_centroid

    def extract_spectral_rolloff(self, audio_signal=None, sample_rate=None, n_fft=2048, hop_length=512,
                                 roll_percent=0.85, sample_start=0, sample_end=None):
        if sample_end is None:
            sample_end = self.num_samples
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal[sample_start:sample_end], sr=sample_rate,
                                                            n_fft=n_fft, hop_length=hop_length,
                                                            roll_percent=roll_percent)
        return spectral_rolloff

    def extract_spectral_bandwidth(self, audio_signal=None, sample_rate=None, n_fft=2048, hop_length=512, p=2,
                                   sample_start=0, sample_end=None):
        if sample_end is None:
            sample_end = self.num_samples
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_signal[sample_start:sample_end], sr=sample_rate,
                                                                n_fft=n_fft, hop_length=hop_length, p=p)
        return spectral_bandwidth

    def extract_spectral_flux(self, audio_signal=None, sample_rate=None, sample_start=0, sample_end=None):
        if sample_end is None:
            sample_end = self.num_samples
        spectral_flux = librosa.feature.spectral_bandwidth(y=audio_signal[sample_start:sample_end], sr=sample_rate)
        return spectral_flux

    def create_wave_plot(self, audio_signal=None, sample_rate=None, sample_start=0, ax=None):

        wave_plot = lib_disp.waveplot(audio_signal[sample_start:self.num_samples], sr=sample_rate, alpha=0.5,
                                      x_axis='time', ax=ax)
        return wave_plot

    def create_magnitude_plot(self, audio_signal=None, sample_rate=None, sample_start=0, sample_end=None, f_ratio=0.5,
                              ax=None):
        if sample_end is None:
            sample_end = self.num_samples

        sp = np.fft.fft(audio_signal[sample_start:sample_end])
        mag = np.absolute(sp)
        f = np.linspace(0, sample_rate, len(mag))
        f_bins = int(len(mag) * f_ratio)

        ax.set_xlabel('Hz')
        magnitude_plot = ax.plot(f[:f_bins], mag[:f_bins])

        return magnitude_plot

    def create_spectrogram(self, audio_signal=None, sample_rate=None, sample_start=0,
                           sample_end=None, ax=None):
        if sample_end is None:
            sample_end = self.num_samples
        audio_stft = librosa.stft(audio_signal[sample_start:sample_end])
        audio_db = librosa.amplitude_to_db(np.abs(audio_stft), ref=np.max)
        spectrogram = lib_disp.specshow(audio_db, sr=sample_rate, y_axis='linear', x_axis='time', ax=ax)
        return spectrogram

    def create_melspec_plot(self, audio_signal=None, sample_rate=None, n_fft=2048, hop_length=512,
                            sample_start=0,
                            sample_end=None, ax=None):
        if sample_end is None:
            sample_end = self.num_samples
        mel = librosa.feature.melspectrogram(audio_signal[sample_start:sample_end], sr=sample_rate, n_fft=n_fft,
                                             hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        melspec_plot = lib_disp.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax)

        return melspec_plot

    def create_mfcc_plot(self, audio_signal=None, sample_rate=None,
                         n_fft=2048, hop_length=512, n_mfcc=20, sample_start=0, sample_end=None, ax=None):
        if sample_end is None:
            sample_end = self.num_samples
        mfcc = librosa.feature.mfcc(audio_signal[sample_start:sample_end],
                                    sr=sample_rate,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        mfcc_plot = lib_disp.specshow(mfcc, sr=sample_rate, x_axis='time', ax=ax)
        return mfcc_plot
