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

    def __init__(self, filename=None,
                 sample_rate=22050,
                 time_start=0,
                 duration=29,
                 frame_length=2048,
                 n_fft=2048,
                 hop_length=512):
        self._filename = os.path.abspath(os.path.expanduser(filename))
        self._sample_rate = sample_rate
        self._time_start = time_start
        self._duration = duration
        self._audio_signal = None
        self._sample_start = None
        self._sample_end = None
        self._frame_length = frame_length
        self._n_fft = n_fft
        self._hop_length = hop_length

    @property
    def filename(self):
        return self._filename

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, new_rate):
        self._sample_rate = new_rate
        self._audio_signal = None
        self._sample_start = None
        self._sample_end = None

    @sample_rate.deleter
    def sample_rate(self):
        self._sample_rate = None
        self._audio_signal = None
        self._sample_start = None
        self._sample_end = None

    @property
    def audio_signal(self):
        if self._audio_signal is None:
            self._audio_signal, _ = self.__load_wav_file()
        return self._audio_signal

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, new_time):
        self._time_start = new_time
        self._sample_start = None
        self._sample_end = None

    @time_start.deleter
    def time_start(self):
        self._time_start = None
        self._sample_start = None
        self._sample_end = None

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, new_duration):
        self._duration = new_duration
        self._sample_end = None

    @duration.deleter
    def duration(self):
        self._duration = None
        self._sample_end = None

    @property
    def sample_start(self):
        if self._sample_start is None:
            self._sample_start = self.time_start * self.sample_rate
        return self._sample_start

    @property
    def sample_end(self):
        if self._sample_end is None:
            self._sample_end = self.sample_start + self.sample_rate * self.duration
        return self._sample_end

    @property
    def frame_length(self):
        return self._frame_length

    @frame_length.setter
    def frame_length(self, new_length):
        self._frame_length = new_length

    @frame_length.deleter
    def frame_length(self):
        self._frame_length = None

    @property
    def n_fft(self):
        return self._n_fft

    @n_fft.setter
    def n_fft(self, new_n_fft):
        self._n_fft = new_n_fft

    @n_fft.deleter
    def n_fft(self):
        self._n_fft = None

    @property
    def hop_length(self):
        return self._hop_length

    @hop_length.setter
    def hop_length(self, new_hop_length):
        self._hop_length = new_hop_length

    @hop_length.deleter
    def hop_length(self):
        self._hop_length = None

    def __convert_mp3_to_wav(self):
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

    def __load_wav_file(self):
        if ".wav" in self.filename:
            wav_file = self.filename
        elif ".mp3" in self.filename:
            wav_file = self.__convert_mp3_to_wav()
        else:
            raise Exception("File could not be loaded.")

        audio_signal, sample_rate = librosa.load(wav_file, sr=self.sample_rate)
        return audio_signal, sample_rate

    def extract_rsm(self):
        rms = librosa.feature.rms(y=self.audio_signal[self.sample_start:self.sample_end],
                                  frame_length=self.frame_length,
                                  hop_length=self.hop_length)
        return rms

    def extract_zero_crossing_rate(self):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=self.audio_signal[self.sample_start:self.sample_end],
                                                                frame_length=self.frame_length,
                                                                hop_length=self.hop_length)
        return zero_crossing_rate

    def extract_spectral_centroid(self):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio_signal[self.sample_start:self.sample_end],
                                                              sr=self.sample_rate, n_fft=self.n_fft,
                                                              hop_length=self.hop_length)
        return spectral_centroid

    def extract_spectral_rolloff(self, roll_percent):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio_signal[self.sample_start:self.sample_end],
                                                            sr=self.sample_rate, n_fft=self.n_fft,
                                                            hop_length=self.hop_length, roll_percent=roll_percent)
        return spectral_rolloff

    def extract_spectral_bandwidth(self, p=2):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.audio_signal[self.sample_start:self.sample_end],
                                                                sr=self.sample_rate, n_fft=self.n_fft,
                                                                hop_length=self.hop_length, p=p)
        return spectral_bandwidth

    def extract_spectral_flux(self):
        spectral_flux = librosa.feature.spectral_bandwidth(y=self.audio_signal[self.sample_start:self.sample_end],
                                                           sr=self.sample_rate)
        return spectral_flux

    def create_wave_plot(self, ax=None):

        wave_plot = lib_disp.waveshow(y=self.audio_signal[self.sample_start:self.sample_end],
                                      sr=self.sample_rate, alpha=0.5, x_axis='time', ax=ax)
        return wave_plot

    def create_magnitude_plot(self, f_ratio=0.5, ax=None):
        sp = np.fft.fft(self.audio_signal[self.sample_start:self.sample_end])
        mag = np.absolute(sp)
        f = np.linspace(0, self.sample_rate, len(mag))
        f_bins = int(len(mag) * f_ratio)

        ax.set_xlabel('Hz')
        magnitude_plot = ax.plot(f[:f_bins], mag[:f_bins])

        return magnitude_plot

    def create_spectrogram(self, ax=None):
        audio_stft = librosa.stft(y=self.audio_signal[self.sample_start:self.sample_end])
        audio_db = librosa.amplitude_to_db(np.abs(audio_stft), ref=np.max)
        spectrogram = lib_disp.specshow(audio_db, sr=self.sample_rate, y_axis='linear', x_axis='time', ax=ax)
        return spectrogram

    def create_melspec_plot(self, ax=None):
        mel = librosa.feature.melspectrogram(y=self.audio_signal[self.sample_start:self.sample_end],
                                             sr=self.sample_rate, n_fft=self.n_fft,
                                             hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        melspec_plot = lib_disp.specshow(mel_db, sr=self.sample_rate, x_axis='time', y_axis='mel', ax=ax)

        return melspec_plot

    def create_mfcc_plot(self, n_mfcc=20, ax=None):
        mfcc = librosa.feature.mfcc(y=self.audio_signal[self.sample_start:self.sample_end],
                                    sr=self.sample_rate,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    n_mfcc=n_mfcc)
        mfcc_plot = lib_disp.specshow(mfcc, sr=self.sample_rate, x_axis='time', ax=ax)
        return mfcc_plot
