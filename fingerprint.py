"""
fingerprint.py
Mô-đun cốt lõi Xử lý Tín hiệu Số (DSP):
1. Tính biến đổi Fourier thời gian ngắn (STFT - Short-Time Fourier Transform).
2. Xây dựng bản đồ phổ biên độ Spectrogram và áp dụng cửa sổ Hanning.
3. Trích xuất các điểm cực đại biên độ cục bộ (2D Peak Finding / Constellation Map).
4. Ghép cặp các đỉnh tạo mã băm Landmark (Shazam Fingerprint Hashing).
"""

import numpy as np
from scipy import fft
from scipy.ndimage import maximum_filter

# Các hằng số DSP chuẩn hóa
N_FFT = 1024           # Kích thước khung FFT (độ dài N)
HOP_LENGTH = 512       # Bước nhảy giữa các khung (50% Overlap)
DEFAULT_FS = 22050     # Tần số lấy mẫu mặc định (Hz)

# Tham số vùng mục tiêu cho thuật toán ghép cặp Landmark (Target Zone)
DELTA_T_MIN = 2        # Khoảng cách tối thiểu giữa 2 đỉnh (khung)
DELTA_T_MAX = 35       # Khoảng cách tối đa nhìn về tương lai (khung) ~ 0.8 giây
MAX_PAIRS_PER_PEAK = 5 # Số lượng cặp tối đa cho mỗi điểm neo (Anchor point)

# Tham số lọc cực đại cục bộ 2 chiều (2D Peak Extraction)
FILTER_SIZE = (9, 9)   # Kích thước lân cận (9 bin tần số x 9 khung thời gian)
THRESHOLD_REL = -25.0  # Ngưỡng năng lượng tương đối (dB) so với đỉnh lớn nhất

def compute_stft(audio: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                 window_type: str = 'hann') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính biến đổi Fourier thời gian ngắn (STFT) cho tín hiệu rời rạc x[n].
    
    Các bước DSP:
    1. Chia tín hiệu x[n] thành các khung có độ dài n_fft với độ chồng lấp (n_fft - hop_length).
    2. Nhân mỗi khung với hàm cửa sổ làm trơn (Hanning Window) để giảm rò rỉ phổ:
       w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    3. Thực hiện biến đổi FFT thực (rfft) cho từng khung:
       X_m[k] = sum_{n=0}^{N-1} x_m[n] * exp(-j * 2*pi * k * n / N)
    
    Trả về:
    - spectrogram_db: Ma trận phổ năng lượng (dB), kích thước (Số bin tần số, Số khung thời gian).
    - freqs: Mảng tần số tương ứng của các bin (Hz).
    - times: Mảng thời gian tương ứng của các khung (giây).
    """
    num_samples = len(audio)
    num_frames = 1 + (num_samples - n_fft) // hop_length
    
    if num_frames <= 0:
        # Nếu tín hiệu quá ngắn, đệm thêm số 0 (Zero-padding)
        pad_len = n_fft - num_samples
        audio = np.pad(audio, (0, pad_len), mode='constant')
        num_frames = 1
        
    # Tạo cửa sổ Hanning
    if window_type == 'hann':
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
    else:
        window = np.ones(n_fft)
        
    # Tạo ma trận khung (Framing)
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)
    for m in range(num_frames):
        start = m * hop_length
        frames[m] = audio[start:start + n_fft] * window
        
    # Thực hiện rfft trên từng khung
    # Kết quả phổ có (n_fft // 2 + 1) bin tần số (từ 0 đến fs / 2 theo định lý Nyquist)
    stft_matrix = fft.rfft(frames, n=n_fft, axis=1) # (num_frames, num_bins)
    
    # Tính phổ biên độ |X_m[k]|
    magnitude = np.abs(stft_matrix).T # chuyển vị thành (num_bins, num_frames)
    
    # Chuyển đổi sang thang đo Decibel (dB): S_dB = 20 * log10(|X| + epsilon)
    epsilon = 1e-6
    spectrogram_db = 20.0 * np.log10(magnitude + epsilon)
    
    # Tính vector tần số và thời gian
    freq_bins = n_fft // 2 + 1
    freqs = np.linspace(0, DEFAULT_FS / 2, freq_bins)
    times = (np.arange(num_frames) * hop_length) / DEFAULT_FS
    
    return spectrogram_db, freqs, times

def extract_peaks(spectrogram_db: np.ndarray, threshold_rel: float = THRESHOLD_REL,
                  filter_size: tuple[int, int] = FILTER_SIZE) -> list[tuple[int, int]]:
    """
    Trích xuất các điểm cực đại cục bộ trong ma trận phổ 2D (Constellation Map).
    
    Nguyên lý DSP:
    - Áp dụng bộ lọc cực đại cục bộ 2 chiều (2D Maximum Filter) với kích thước lân cận (filter_size).
    - Ngưỡng biên độ tương đối (threshold_rel): Chỉ giữ lại các đỉnh có năng lượng cách đỉnh
      lớn nhất của toàn bản nhạc không quá |threshold_rel| dB, giúp loại bỏ nhiễu trắng nền
      và bảo toàn các đỉnh nhạc chính.
    
    Trả về:
    - Danh sách các đỉnh: list các tuple (time_idx, freq_idx) đã sắp xếp theo thời gian.
    """
    max_val = np.max(spectrogram_db)
    amp_threshold = max_val + threshold_rel
    
    # Bộ lọc cực đại 2 chiều
    local_max = maximum_filter(spectrogram_db, size=filter_size)
    
    # Điều kiện cực đại cục bộ và vượt ngưỡng
    detected_peaks = (spectrogram_db == local_max) & (spectrogram_db >= amp_threshold)
    
    # Lấy tọa độ (freq_idx, time_idx)
    freq_indices, time_indices = np.where(detected_peaks)
    
    # Sắp xếp các đỉnh theo trục thời gian tăng dần
    peaks = list(zip(time_indices, freq_indices))
    peaks.sort(key=lambda p: (p[0], p[1]))
    
    return peaks

def generate_landmark_hashes(peaks: list[tuple[int, int]], song_id: int = None,
                             delta_t_min: int = DELTA_T_MIN,
                             delta_t_max: int = DELTA_T_MAX,
                             max_pairs: int = MAX_PAIRS_PER_PEAK) -> list[tuple[str, int, int]]:
    """
    Thuật toán ghép cặp Landmark Hashing (Shazam Algorithm):
    
    Với mỗi điểm cực đại neo (Anchor Peak) tại thời điểm t1 và tần số f1:
    - Mở một vùng mục tiêu (Target Zone) về phía trước trục thời gian:
      t1 + delta_t_min <= t2 <= t1 + delta_t_max
    - Ghép cặp (f1, f2, delta_t) thành một chuỗi mã băm đại diện.
    - Thời gian xuất hiện của mã băm này là t1 (time offset).
    
    Cấu trúc mã băm: "f1|f2|delta_t"
    
    Trả về:
    - Danh sách các tuple: (hash_str, t1_offset, song_id)
    """
    hashes = []
    num_peaks = len(peaks)
    
    for i in range(num_peaks):
        t1, f1 = peaks[i]
        pairs_found = 0
        
        for j in range(i + 1, num_peaks):
            t2, f2 = peaks[j]
            delta_t = t2 - t1
            
            # Nếu đã vượt quá thời gian tối đa nhìn về tương lai thì dừng xét đỉnh neo này
            if delta_t > delta_t_max:
                break
                
            # Đảm bảo thỏa mãn khoảng cách tối thiểu (tránh ghép cặp đỉnh của cùng nốt)
            if delta_t >= delta_t_min:
                hash_key = f"{f1}|{f2}|{delta_t}"
                hashes.append((hash_key, t1, song_id))
                pairs_found += 1
                
                if pairs_found >= max_pairs:
                    break
                    
    return hashes

def fingerprint_audio(audio: np.ndarray, song_id: int = None) -> tuple[list[tuple[str, int, int]], list[tuple[int, int]], np.ndarray]:
    """
    Hàm đóng gói quy trình trích xuất dấu vân tay âm thanh hoàn chỉnh:
    Audio -> STFT -> Spectrogram -> Peak Extraction -> Landmark Hashes.
    
    Trả về:
    - hashes: Danh sách mã băm (hash_key, t1_offset, song_id)
    - peaks: Danh sách các đỉnh phổ (time_idx, freq_idx)
    - spectrogram_db: Ma trận phổ năng lượng
    """
    spec_db, freqs, times = compute_stft(audio)
    peaks = extract_peaks(spec_db)
    hashes = generate_landmark_hashes(peaks, song_id=song_id)
    return hashes, peaks, spec_db
