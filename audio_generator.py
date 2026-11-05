"""
audio_generator.py
Mô-đun tổng hợp tín hiệu âm thanh mẫu bằng các kỹ thuật DSP (Additive Synthesis, ADSR Envelope).
Dùng để tạo dữ liệu âm thanh kiểm thử cho hệ thống nhận diện bài hát Mini-Shazam.
"""

import os
import numpy as np
import scipy.io.wavfile as wav

SAMPLE_RATE = 22050  # Tần số lấy mẫu chuẩn (Hz)

def note_to_freq(note_name: str) -> float:
    """
    Chuyển đổi tên nốt nhạc (ví dụ: 'C4', 'A4', 'G#5') thành tần số (Hz).
    Công thức tiêu chuẩn quốc tế: f = 440 * 2^((midi - 69) / 12)
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    name = note_name[:-1].upper()
    octave = int(note_name[-1])
    semitone = notes.index(name)
    midi_num = semitone + (octave + 1) * 12
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))

def generate_tone(freq: float, duration: float, sample_rate: int = SAMPLE_RATE,
                  envelope_type: str = 'piano') -> np.ndarray:
    """
    Tổng hợp một nốt nhạc bằng phương pháp Additive Synthesis với họa âm và bao biên ADSR.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if freq <= 0:
        return np.zeros_like(t)
    
    # Tổng hợp các họa âm (Harmonics) để tạo âm sắc tự nhiên
    # Họa âm bậc 1 (cơ bản), bậc 2, bậc 3, bậc 4
    signal = (
        1.00 * np.sin(2 * np.pi * freq * t) +
        0.50 * np.sin(2 * np.pi * 2 * freq * t) +
        0.25 * np.sin(2 * np.pi * 3 * freq * t) +
        0.12 * np.sin(2 * np.pi * 4 * freq * t)
    )
    
    # Thiết kế bao biên biên độ ADSR (Attack, Decay, Sustain, Release)
    n_samples = len(t)
    envelope = np.ones(n_samples)
    
    if envelope_type == 'piano':
        # Piano: Attack nhanh, Decay dài, suy giảm dần theo hàm mũ
        attack_len = int(0.02 * sample_rate)
        if attack_len > 0 and attack_len < n_samples:
            envelope[:attack_len] = np.linspace(0, 1, attack_len)
            decay_curve = np.exp(-3.5 * np.linspace(0, 1, n_samples - attack_len))
            envelope[attack_len:] = decay_curve
    elif envelope_type == 'pluck':
        # Tiếng gảy dây đàn: suy giảm nhanh
        decay_curve = np.exp(-6.0 * np.linspace(0, 1, n_samples))
        envelope = decay_curve
        
    return signal * envelope

def synthesize_melody(notes_list, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Ghép các nốt nhạc thành một bài nhạc hoàn chỉnh.
    notes_list: Danh sách tuple (note_name, duration_seconds)
    """
    track = []
    for note, dur in notes_list:
        if note == 'REST':
            chunk = np.zeros(int(sample_rate * dur))
        else:
            freq = note_to_freq(note)
            chunk = generate_tone(freq, dur, sample_rate=sample_rate)
        track.append(chunk)
    
    audio = np.concatenate(track)
    # Chuẩn hóa về biên độ [-1.0, 1.0]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def add_awgn_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Thêm nhiễu trắng Gaussian (AWGN) vào tín hiệu với tỉ số tín hiệu trên nhiễu SNR (dB).
    SNR_dB = 10 * log10(P_signal / P_noise)
    """
    sig_power = np.mean(signal ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    noisy_signal = signal + noise
    # Chuẩn hóa tránh tràn số
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 0:
        noisy_signal = noisy_signal / max_val
    return noisy_signal

def save_wav(filename: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Lưu mảng numpy float [-1.0, 1.0] thành file WAV 16-bit PCM."""
    int16_audio = np.int16(audio * 32767)
    wav.write(filename, sample_rate, int16_audio)

def load_audio(filename: str, target_sr: int = SAMPLE_RATE) -> tuple[int, np.ndarray]:
    """
    Đọc file âm thanh hỗ trợ đa định dạng (.wav, .mp3, .m4a, .flac, .ogg).
    - Tự động chuyển đổi sang kênh đơn (Mono).
    - Tự động lấy mẫu lại (Resample) về tần số chuẩn target_sr (mặc định 22050 Hz).
    - Chuẩn hóa biên độ tín hiệu về miền [-1.0, 1.0].
    Sử dụng kết hợp ffmpeg (pipe trực tiếp) và scipy.io.wavfile làm fallback.
    """
    import subprocess
    ext = os.path.splitext(filename)[1].lower()
    
    # 1. Ưu tiên giải mã trực tiếp bằng ffmpeg (hỗ trợ hoàn hảo MP3, M4A, FLAC, WAV, v.v.)
    cmd = [
        'ffmpeg', '-nostdin', '-loglevel', 'error',
        '-i', filename,
        '-ac', '1',              # Ép về 1 kênh Mono
        '-ar', str(target_sr),   # Resample về tần số chuẩn
        '-f', 'f32le',           # Xuất định dạng 32-bit float Little Endian
        'pipe:1'
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and len(proc.stdout) > 0:
            audio = np.frombuffer(proc.stdout, dtype=np.float32)
            # Chuẩn hóa biên độ nếu cần
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            return target_sr, audio
    except Exception:
        pass
        
    # 2. Fallback cho file WAV thuần nếu không gọi được ffmpeg
    if ext == '.wav':
        sr, data = wav.read(filename)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128) / 128.0
        else:
            audio = data.astype(np.float32)
            
        # Nếu sample rate của file khác target_sr, thực hiện nội suy đơn giản
        if sr != target_sr:
            num_target_samples = int(len(audio) * target_sr / sr)
            audio = np.interp(
                np.linspace(0, len(audio), num_target_samples, endpoint=False),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)
            sr = target_sr
            
        return sr, audio
        
    raise RuntimeError(f"Không thể đọc file âm thanh '{filename}'. Hãy đảm bảo ffmpeg đã được cài đặt.")

# Alias tương thích ngược với code cũ
load_wav = load_audio

def generate_sample_database(output_dir: str = 'data'):
    """
    Tạo cơ sở dữ liệu gồm 4 bài hát mẫu kinh điển và 1 bài kiểm thử ngoại lai (Unregistered song).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Canon in D (Pachelbel) - Giai điệu mượt mà, nhiều nốt hợp âm
    canon_in_d = [
        ('F#4', 0.5), ('E4', 0.5), ('D4', 0.5), ('C#4', 0.5),
        ('B3', 0.5), ('A3', 0.5), ('B3', 0.5), ('C#4', 0.5),
        ('D4', 0.5), ('F#4', 0.5), ('A4', 0.5), ('G4', 0.5),
        ('F#4', 0.5), ('D4', 0.5), ('F#4', 0.5), ('E4', 0.5),
        ('D4', 0.25), ('C#4', 0.25), ('D4', 0.25), ('E4', 0.25),
        ('F#4', 0.5), ('E4', 0.5), ('D4', 0.5), ('C#4', 0.5),
        ('B3', 0.5), ('A3', 0.5), ('B3', 0.5), ('C#4', 0.5),
        ('D4', 1.0)
    ]
    
    # 2. Für Elise (Beethoven) - Cao độ thay đổi nhanh, tiết tấu rõ
    fur_elise = [
        ('E5', 0.3), ('D#5', 0.3), ('E5', 0.3), ('D#5', 0.3), ('E5', 0.3),
        ('B4', 0.3), ('D5', 0.3), ('C5', 0.3), ('A4', 0.6),
        ('REST', 0.2),
        ('C4', 0.3), ('E4', 0.3), ('A4', 0.3), ('B4', 0.6),
        ('REST', 0.2),
        ('E4', 0.3), ('G#4', 0.3), ('B4', 0.3), ('C5', 0.6),
        ('REST', 0.2),
        ('E4', 0.3), ('E5', 0.3), ('D#5', 0.3), ('E5', 0.3), ('D#5', 0.3),
        ('E5', 0.3), ('B4', 0.3), ('D5', 0.3), ('C5', 0.3), ('A4', 0.8)
    ]
    
    # 3. Happy Birthday - Quen thuộc, cao độ biến đổi rộng
    happy_birthday = [
        ('C4', 0.35), ('C4', 0.15), ('D4', 0.5), ('C4', 0.5), ('F4', 0.5), ('E4', 1.0),
        ('C4', 0.35), ('C4', 0.15), ('D4', 0.5), ('C4', 0.5), ('G4', 0.5), ('F4', 1.0),
        ('C4', 0.35), ('C4', 0.15), ('C5', 0.5), ('A4', 0.5), ('F4', 0.5), ('E4', 0.5), ('D4', 0.5),
        ('A#4', 0.35), ('A#4', 0.15), ('A4', 0.5), ('F4', 0.5), ('G4', 0.5), ('F4', 1.2)
    ]
    
    # 4. Twinkle Twinkle Little Star - Âm sắc đều, nốt lặp
    twinkle_star = [
        ('C4', 0.5), ('C4', 0.5), ('G4', 0.5), ('G4', 0.5),
        ('A4', 0.5), ('A4', 0.5), ('G4', 1.0),
        ('F4', 0.5), ('F4', 0.5), ('E4', 0.5), ('E4', 0.5),
        ('D4', 0.5), ('D4', 0.5), ('C4', 1.0),
        ('G4', 0.5), ('G4', 0.5), ('F4', 0.5), ('F4', 0.5),
        ('E4', 0.5), ('E4', 0.5), ('D4', 1.0)
    ]
    
    # 5. Bài hát ngoại lai (Unregistered Song) dùng để kiểm tra chống nhận diện nhầm:
    # "Mission Impossible Riff" (Ngoại lai - KHÔNG đưa vào Database)
    unregistered_song = [
        ('G4', 0.4), ('G4', 0.4), ('A#4', 0.3), ('C5', 0.3),
        ('G4', 0.4), ('G4', 0.4), ('F4', 0.3), ('F#4', 0.3),
        ('G4', 0.4), ('G4', 0.4), ('A#4', 0.3), ('C5', 0.3),
        ('G4', 0.4), ('G4', 0.4), ('F4', 0.3), ('F#4', 0.3)
    ]
    
    songs = {
        'Pachelbel_Canon_in_D': canon_in_d,
        'Beethoven_Fur_Elise': fur_elise,
        'Happy_Birthday_Melody': happy_birthday,
        'Twinkle_Little_Star': twinkle_star,
        'Unknown_Unregistered_Song': unregistered_song
    }
    
    created_files = {}
    for name, notes in songs.items():
        audio = synthesize_melody(notes, sample_rate=SAMPLE_RATE)
        path = os.path.join(output_dir, f"{name}.wav")
        save_wav(path, audio, sample_rate=SAMPLE_RATE)
        created_files[name] = path
        
    return created_files

if __name__ == '__main__':
    print("Đang khởi tạo các bài hát mẫu bằng kỹ thuật DSP Additive Synthesis...")
    files = generate_sample_database()
    for name, path in files.items():
        print(f"  + Đã tạo: {name} -> {path}")
