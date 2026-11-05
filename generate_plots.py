"""
generate_plots.py
Tạo các biểu đồ trực quan hóa chuyên sâu cho Báo cáo bài tập lớn Xử lý Tín hiệu Số (DSP):
1. Dạng sóng và Bản đồ phổ thời gian - tần số (Spectrogram 2D).
2. Bản đồ chòm sao các đỉnh cực đại (Constellation Map).
3. Nguyên lý ghép cặp Landmark Hashing (Target Zone).
4. Phân tích độ lệch thời gian Histogram giữa bài hát đúng và sai.
5. Thử nghiệm độ bền vững trước nhiễu AWGN ở nhiều mức SNR khác nhau.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Chế độ headless render
import matplotlib.pyplot as plt
import numpy as np

from audio_generator import load_wav, add_awgn_noise, SAMPLE_RATE
from fingerprint import compute_stft, extract_peaks, generate_landmark_hashes, HOP_LENGTH, DEFAULT_FS
from recognizer import MiniShazamRecognizer

OUTPUT_DIR = 'output'
DATA_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_stft_and_peaks():
    """Vẽ dạng sóng, Spectrogram và Constellation Map cho bài Für Elise."""
    file_path = os.path.join(DATA_DIR, 'Beethoven_Fur_Elise.wav')
    sr, audio = load_wav(file_path)
    
    duration = min(6.0, len(audio) / sr)
    audio_clip = audio[:int(duration * sr)]
    
    spec_db, freqs, times = compute_stft(audio_clip)
    peaks = extract_peaks(spec_db)
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    
    # 1. Dạng sóng miền thời gian
    t_audio = np.linspace(0, duration, len(audio_clip))
    axes[0].plot(t_audio, audio_clip, color='#1f77b4', linewidth=0.8)
    axes[0].set_title("(a) Tín hiệu âm thanh miền thời gian x[n] (Beethoven - Für Elise)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Biên độ chuẩn hóa", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Phổ Spectrogram (STFT 2D)
    freq_mask = freqs <= 3500
    im = axes[1].pcolormesh(times, freqs[freq_mask], spec_db[freq_mask, :], 
                            shading='gouraud', cmap='magma')
    axes[1].set_title("(b) Bản đồ phổ biên độ Spectrogram (STFT với Cửa sổ Hanning)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Tần số (Hz)", fontsize=10)
    fig.colorbar(im, ax=axes[1], format="%+2.0f dB", pad=0.02, label='Biên độ (dB)')
    
    # 3. Constellation Map
    axes[2].pcolormesh(times, freqs[freq_mask], spec_db[freq_mask, :], 
                       shading='gouraud', cmap='viridis', alpha=0.35)
    
    peak_times = [times[p[0]] for p in peaks if p[0] < len(times) and freqs[p[1]] <= 3500]
    peak_freqs = [freqs[p[1]] for p in peaks if p[0] < len(times) and freqs[p[1]] <= 3500]
    
    axes[2].scatter(peak_times, peak_freqs, color='#e74c3c', s=35, edgecolors='black', 
                    linewidth=0.6, label=f'Đỉnh cực đại (N = {len(peak_times)})')
    axes[2].set_title("(c) Bản đồ chòm sao (Constellation Map - 2D Local Maximum Peaks)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Thời gian (giây)", fontsize=10)
    axes[2].set_ylabel("Tần số (Hz)", fontsize=10)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'plot_1_stft_and_peaks.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Đã lưu: {out_path}")

def plot_landmark_pairs():
    """Minh họa trực quan thuật toán ghép cặp Landmark Hashing."""
    file_path = os.path.join(DATA_DIR, 'Beethoven_Fur_Elise.wav')
    sr, audio = load_wav(file_path)
    audio_clip = audio[:int(3.0 * sr)]
    
    spec_db, freqs, times = compute_stft(audio_clip)
    peaks = extract_peaks(spec_db)
    
    valid_peaks = [p for p in peaks if p[0] < len(times) and freqs[p[1]] <= 3000]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    freq_mask = freqs <= 3000
    ax.pcolormesh(times, freqs[freq_mask], spec_db[freq_mask, :], shading='gouraud', cmap='Blues', alpha=0.25)
    
    peak_times = [times[p[0]] for p in valid_peaks]
    peak_freqs = [freqs[p[1]] for p in valid_peaks]
    ax.scatter(peak_times, peak_freqs, color='#34495e', s=40, zorder=3, label='Các đỉnh cực đại')
    
    # Chọn một điểm neo nổi bật
    if len(valid_peaks) >= 6:
        anchor_idx = 2
        anchor_peak = valid_peaks[anchor_idx]
        t1_idx, f1_idx = anchor_peak
        t1_time = times[t1_idx]
        f1_freq = freqs[f1_idx]
        
        ax.scatter([t1_time], [f1_freq], color='#e74c3c', s=130, edgecolors='black', 
                   linewidth=1.5, zorder=5, label='Điểm neo (Anchor Peak - f1, t1)')
        
        target_count = 0
        for j in range(anchor_idx + 1, len(valid_peaks)):
            t2_idx, f2_idx = valid_peaks[j]
            delta_t = t2_idx - t1_idx
            if delta_t > 35:
                break
            if delta_t >= 2:
                t2_time = times[t2_idx]
                f2_freq = freqs[f2_idx]
                ax.plot([t1_time, t2_time], [f1_freq, f2_freq], color='#e67e22', 
                        linestyle='-', linewidth=1.8, alpha=0.85, zorder=4)
                ax.scatter([t2_time], [f2_freq], color='#27ae60', s=80, edgecolors='black', 
                           linewidth=1.0, zorder=5)
                target_count += 1
                if target_count >= 5:
                    break
                    
        ax.scatter([], [], color='#27ae60', s=80, edgecolors='black', 
                   label='Các đỉnh ghép đôi trong Target Zone (f2, t2)')
        ax.plot([], [], color='#e67e22', linestyle='-', linewidth=1.8, 
                label='Mã băm Landmark [f1 | f2 | Δt]')
        
    ax.set_title("Nguyên lý Ghép cặp Điểm neo & Vùng mục tiêu (Landmark Combinatorial Hashing)", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Thời gian (giây)", fontsize=11)
    ax.set_ylabel("Tần số (Hz)", fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'plot_2_landmark_hashing.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Đã lưu: {out_path}")

def plot_noise_comparison():
    """So sánh tín hiệu sạch và tín hiệu bị nhiễm nhiễu AWGN SNR = 5 dB."""
    file_path = os.path.join(DATA_DIR, 'Beethoven_Fur_Elise.wav')
    sr, clean_audio = load_wav(file_path)
    clean_clip = clean_audio[:int(4.0 * sr)]
    
    noisy_clip = add_awgn_noise(clean_clip, snr_db=5.0)
    
    spec_clean, freqs, times = compute_stft(clean_clip)
    spec_noisy, _, _ = compute_stft(noisy_clip)
    
    peaks_clean = extract_peaks(spec_clean)
    peaks_noisy = extract_peaks(spec_noisy)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    freq_mask = freqs <= 3500
    
    # 1. Dạng sóng sạch
    t = np.linspace(0, 4.0, len(clean_clip))
    axes[0, 0].plot(t, clean_clip, color='#2980b9', linewidth=0.8)
    axes[0, 0].set_title("Tín hiệu gốc sạch (Clean Audio)", fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel("Biên độ", fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.4)
    
    # 2. Dạng sóng dính nhiễu
    axes[0, 1].plot(t, noisy_clip, color='#c0392b', linewidth=0.8)
    axes[0, 1].set_title("Tín hiệu lẫn tạp âm (AWGN SNR = 5 dB)", fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.4)
    
    # 3. Spectrogram sạch + Peaks
    axes[1, 0].pcolormesh(times, freqs[freq_mask], spec_clean[freq_mask, :], 
                          shading='gouraud', cmap='inferno')
    p_clean_times = [times[p[0]] for p in peaks_clean if p[0] < len(times) and freqs[p[1]] <= 3500]
    p_clean_freqs = [freqs[p[1]] for p in peaks_clean if p[0] < len(times) and freqs[p[1]] <= 3500]
    axes[1, 0].scatter(p_clean_times, p_clean_freqs, color='#00ffcc', s=35, edgecolors='black', linewidth=0.6)
    axes[1, 0].set_title(f"Spectrogram Gốc ({len(p_clean_times)} đỉnh)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Thời gian (giây)", fontsize=10)
    axes[1, 0].set_ylabel("Tần số (Hz)", fontsize=10)
    
    # 4. Spectrogram dính nhiễu + Peaks
    axes[1, 1].pcolormesh(times, freqs[freq_mask], spec_noisy[freq_mask, :], 
                          shading='gouraud', cmap='inferno')
    p_noisy_times = [times[p[0]] for p in peaks_noisy if p[0] < len(times) and freqs[p[1]] <= 3500]
    p_noisy_freqs = [freqs[p[1]] for p in peaks_noisy if p[0] < len(times) and freqs[p[1]] <= 3500]
    axes[1, 1].scatter(p_noisy_times, p_noisy_freqs, color='#00ffcc', s=35, edgecolors='black', linewidth=0.6)
    axes[1, 1].set_title(f"Spectrogram Nhiễu ({len(p_noisy_times)} đỉnh - Các đỉnh đặc trưng được bảo toàn)", 
                          fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Thời gian (giây)", fontsize=10)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'plot_3_noise_robustness.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Đã lưu: {out_path}")

def plot_alignment_histogram():
    """
    Minh họa biểu đồ phân bố độ lệch thời gian (Histogram of Delta t)
    chứng minh bài hát đúng có đỉnh nhọn vượt trội, bài hát sai thì phẳng lì.
    """
    rec = MiniShazamRecognizer()
    
    song_files = {
        'Canon in D': os.path.join(DATA_DIR, 'Pachelbel_Canon_in_D.wav'),
        'Für Elise': os.path.join(DATA_DIR, 'Beethoven_Fur_Elise.wav'),
        'Happy Birthday': os.path.join(DATA_DIR, 'Happy_Birthday_Melody.wav'),
        'Twinkle Star': os.path.join(DATA_DIR, 'Twinkle_Little_Star.wav')
    }
    for title, path in song_files.items():
        _, audio = load_wav(path)
        rec.add_song(title, audio)
        
    # Cắt một đoạn truy vấn 3.0 giây từ Für Elise bắt đầu từ giây thứ 1.5 (t = 1.5s)
    _, target_audio = load_wav(song_files['Für Elise'])
    start_sec = 1.5
    duration_sec = 3.0
    start_sample = int(start_sec * SAMPLE_RATE)
    end_sample = start_sample + int(duration_sec * SAMPLE_RATE)
    query_clip = target_audio[start_sample:end_sample]
    
    # Thêm nhiễu nhẹ SNR = 10 dB
    query_clip = add_awgn_noise(query_clip, snr_db=10.0)
    
    result = rec.recognize(query_clip)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram bài đúng
    hist_data = result['best_histogram']
    offsets_sec = [(frame * HOP_LENGTH) / DEFAULT_FS for frame in hist_data.keys()]
    counts = list(hist_data.values())
    
    axes[0].bar(offsets_sec, counts, width=0.15, color='#27ae60', edgecolor='darkgreen', alpha=0.85)
    axes[0].axvline(start_sec, color='#e74c3c', linestyle='--', linewidth=2, 
                    label=f'Vị trí gốc thực tế (t = {start_sec:.1f}s)')
    axes[0].set_title(f"Bài đúng: {result['title']} (Đỉnh nhọn cực đại = {result['peak_score']})", 
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Độ lệch thời gian offset Δt = t_song - t_query (giây)", fontsize=10)
    axes[0].set_ylabel("Số lượng cặp mã băm trùng khớp", fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.4)
    
    # So sánh điểm số giữa tất cả các bài hát trong CSDL
    titles = [s['title'] for s in result['all_scores'].values()]
    scores = [s['peak_score'] for s in result['all_scores'].values()]
    colors = ['#27ae60' if 'Für Elise' in t else '#95a5a6' for t in titles]
    
    axes[1].barh(titles, scores, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title("So sánh điểm số đỉnh (Peak Consensus Score) giữa các bài", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Số lượng mã băm đồng thuận tại đỉnh", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'plot_4_temporal_alignment_histogram.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Đã lưu: {out_path}")

def plot_snr_performance():
    """Khảo sát tỷ lệ nhận diện chính xác theo mức độ nhiễu SNR (từ -5 dB đến 25 dB)."""
    rec = MiniShazamRecognizer(min_peak_score=8, min_confidence=0.20)
    song_files = {
        'Canon in D': os.path.join(DATA_DIR, 'Pachelbel_Canon_in_D.wav'),
        'Für Elise': os.path.join(DATA_DIR, 'Beethoven_Fur_Elise.wav'),
        'Happy Birthday': os.path.join(DATA_DIR, 'Happy_Birthday_Melody.wav'),
        'Twinkle Star': os.path.join(DATA_DIR, 'Twinkle_Little_Star.wav')
    }
    for title, path in song_files.items():
        _, audio = load_wav(path)
        rec.add_song(title, audio)
        
    snr_levels = [-5, 0, 5, 10, 15, 20, 25, 30]
    num_trials = 8
    accuracy = []
    avg_confidences = []
    
    for snr in snr_levels:
        correct = 0
        conf_list = []
        for _ in range(num_trials):
            chosen_title = np.random.choice(list(song_files.keys()))
            _, full_audio = load_wav(song_files[chosen_title])
            
            clip_len = int(3.0 * SAMPLE_RATE)
            max_start = len(full_audio) - clip_len
            start_idx = np.random.randint(0, max_start)
            clip = full_audio[start_idx:start_idx + clip_len]
            
            noisy_clip = add_awgn_noise(clip, snr_db=snr)
            res = rec.recognize(noisy_clip)
            
            if res['match'] and res['title'] == chosen_title:
                correct += 1
                conf_list.append(res['confidence'])
            else:
                conf_list.append(0.0)
                
        accuracy.append((correct / num_trials) * 100)
        avg_confidences.append(np.mean(conf_list) * 100)
        
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    color1 = '#2980b9'
    ax1.set_xlabel("Tỉ số tín hiệu trên nhiễu SNR (dB)", fontsize=11)
    ax1.set_ylabel("Tỉ lệ nhận diện chính xác (%)", color=color1, fontsize=11, fontweight='bold')
    ax1.plot(snr_levels, accuracy, marker='o', linewidth=2.5, markersize=8, color=color1, label='Độ chính xác (%)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = '#e67e22'
    ax2.set_ylabel("Độ tin cậy trung bình (%)", color=color2, fontsize=11, fontweight='bold')
    ax2.plot(snr_levels, avg_confidences, marker='s', linestyle='--', linewidth=2.0, 
             markersize=7, color=color2, label='Độ tin cậy (%)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-5, 105)
    
    plt.title("Đánh giá Độ bền vững của Hệ thống nhận diện dưới các mức nhiễu AWGN", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'plot_5_snr_accuracy_curve.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Đã lưu: {out_path}")

if __name__ == '__main__':
    print("Bắt đầu sinh lại các biểu đồ minh họa DSP...")
    plot_stft_and_peaks()
    plot_landmark_pairs()
    plot_noise_comparison()
    plot_alignment_histogram()
    plot_snr_performance()
    print("Hoàn tất! Toàn bộ biểu đồ đã được cập nhật thành công trong thư mục 'output/'.")
