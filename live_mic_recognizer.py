"""
live_mic_recognizer.py
Chương trình Thu âm Trực tiếp từ Microphone & Nhận diện Bài hát Thời gian Thực (Live Shazam Demo).
Bài tập lớn môn: Xử lý Tín hiệu Số (XLTHS)
Sinh viên thực hiện: Lê Văn Thành - MSV: K25DTCN402
Môn học: Xử lý Tín hiệu Số (DSP).

Cách hoạt động:
1. Nạp toàn bộ các bài hát trong thư mục 'data/' (hỗ trợ .mp3, .wav, .m4a, .flac).
2. Người dùng bật loa phát nhạc (từ điện thoại hoặc trình duyệt) và bấm [Enter] để thu âm.
3. Thu âm trực tiếp qua Micro máy tính (5 giây hoặc tùy chỉnh).
4. Thực hiện thuật toán DSP: STFT -> Spectrogram -> 2D Peak Extraction -> Landmark Hashing.
5. Đối sánh chỉ mục đảo & Phân tích Histogram độ lệch thời gian để xác định bài hát và vị trí giây phát.
"""

import os
import sys
import time
import argparse
import numpy as np

from audio_generator import load_audio, SAMPLE_RATE
from recognizer import MiniShazamRecognizer
from fingerprint import DEFAULT_FS
from main_demo import load_all_songs_from_dir, DATA_DIR

def record_audio(duration: float = 5.0, sample_rate: int = DEFAULT_FS) -> np.ndarray:
    """
    Thu âm từ Micro với thanh tiến trình trực quan.
    Ưu tiên sử dụng sounddevice, tự động fallback sang ffmpeg nếu cần.
    """
    print(f"\n[MIC] Chuẩn bị thu âm trong {duration:.1f} giây...")
    for count in range(3, 0, -1):
        print(f"  >>> Bắt đầu sau: {count}s...", end="\r", flush=True)
        time.sleep(1.0)
    print("  >>> 🔴 ĐANG THU ÂM! Hãy đưa nguồn phát nhạc lại gần micro...             ")
    
    total_samples = int(duration * sample_rate)
    
    # 1. Thu âm bằng sounddevice (Native CoreAudio)
    try:
        import sounddevice as sd
        rec_data = sd.rec(total_samples, samplerate=sample_rate, channels=1, dtype='float32')
        
        # Vẽ thanh tiến trình thu âm
        bar_len = 30
        steps = 20
        step_time = duration / steps
        for i in range(1, steps + 1):
            time.sleep(step_time)
            progress = i / steps
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            curr_sec = i * step_time
            print(f"  [{bar}] {curr_sec:>4.1f}s / {duration:.1f}s", end="\r", flush=True)
            
        sd.wait()
        audio = rec_data.flatten()
        print(f"\n  [✓] Thu âm hoàn tất ({len(audio)} mẫu âm thanh).")
        return audio
    except Exception as e:
        print(f"\n  [!] Sounddevice gặp sự cố ({e}), chuyển sang chế độ dự phòng bằng ffmpeg...")
        
    # 2. Fallback thu âm bằng ffmpeg CLI
    import subprocess
    import tempfile
    temp_wav = tempfile.mktemp(suffix=".wav")
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'avfoundation',
        '-i', ':default',
        '-t', str(duration),
        '-ar', str(sample_rate),
        '-ac', '1',
        temp_wav
    ]
    try:
        subprocess.run(cmd, check=True)
        sr, audio = load_audio(temp_wav)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        print(f"  [✓] Thu âm hoàn tất qua ffmpeg ({len(audio)} mẫu).")
        return audio
    except Exception as e_ff:
        raise RuntimeError(f"Không thể thu âm từ Microphone: {e_ff}")

def run_live_recognition(scan_dir: str = DATA_DIR, record_duration: float = 5.0):
    print("\n" + "=" * 76)
    print("  HỆ THỐNG THU ÂM TRỰC TIẾP TỪ MICRO & NHẬN DIỆN BÀI HÁT (LIVE SHAZAM)")
    print("  Môn học: Xử lý Tín hiệu Số (XLTHS) - Đại học")
    print("=" * 76)
    
    # 1. Khởi tạo Recognizer và nạp các bài hát
    recognizer = MiniShazamRecognizer(min_peak_score=8, min_confidence=0.18)
    loaded_count = load_all_songs_from_dir(scan_dir, recognizer)
    
    if loaded_count == 0:
        print(f"\n[!] Thư mục '{scan_dir}' chưa có file nhạc nào.")
        print("    Hãy copy các file .mp3 hoặc .wav vào thư mục 'data/' rồi chạy lại nhé!")
        return

    while True:
        print("\n" + "-" * 76)
        print("HƯỚNG DẪN:")
        print(f"1. Bật một bài hát trên điện thoại / YouTube (thuộc các bài trong CSDL ở trên).")
        print(f"2. Bấm phím [Enter] trên bàn phím để bắt đầu thu âm {record_duration:.1f} giây.")
        print(f"3. Nhập 'q' rồi bấm [Enter] nếu muốn thoát chương trình.")
        print("-" * 76)
        
        user_choice = input("👉 Nhấn [Enter] để bắt đầu thu âm (hoặc gõ 'q' để thoát): ").strip().lower()
        if user_choice == 'q':
            print("Tạm biệt!")
            break
            
        try:
            # Thu âm từ micro
            audio = record_audio(duration=record_duration, sample_rate=DEFAULT_FS)
            
            # Kiểm tra mức năng lượng (RMS) xem Micro có bắt được tiếng không
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.001:
                print("\n[CẢNH BÁO] Âm lượng thu được quá nhỏ (Micro gần như im lặng)!")
                print("           Hãy kiểm tra quyền Microphone của Terminal hoặc bật nhạc to hơn.")
                
            # Lưu lại file thu âm gần nhất để kiểm tra
            os.makedirs('output', exist_ok=True)
            last_rec_path = os.path.join('output', 'last_live_recording.wav')
            import scipy.io.wavfile as wav
            int16_data = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            wav.write(last_rec_path, DEFAULT_FS, int16_data)
            
            # Chạy thuật toán DSP nhận diện
            print("\n[DSP] Đang xử lý STFT, trích xuất Constellation Peaks và đối sánh Fingerprints...")
            t0 = time.time()
            result = recognizer.recognize(audio)
            proc_time_ms = (time.time() - t0) * 1000
            
            # Hiển thị kết quả
            print("\n" + "=" * 76)
            print("                 KẾT QUẢ NHẬN DIỆN BÀI HÁT")
            print("=" * 76)
            
            if result['match']:
                print(f"  🎶 BÀI HÁT TÌM THẤY   : {result['title']}")
                print(f"  ⏱️  VỊ TRÍ PHÁT TRONG BÀI: Giây thứ {result['offset_sec']:.2f}s")
                print(f"  🎯 ĐIỂM ĐỒNG THUẬN    : {result['peak_score']} mã băm (Landmark Pairs)")
                print(f"  📊 ĐỘ TIN CẬY          : {result['confidence'] * 100:.1f}%")
                print(f"  ⚡ THỜI GIAN NHẬN DIỆN : {proc_time_ms:.2f} ms")
                print(f"  💾 File thu âm đã lưu  : {last_rec_path}")
                print("  => TRẠNG THÁI          : [NHẬN DIỆN THÀNH CÔNG - CHÍNH XÁC]")
            else:
                print(f"  ❌ TRẠNG THÁI          : KHÔNG TÌM THẤY BÀI HÁT PHÙ HỢP")
                print(f"  Lý do                  : {result['title']}")
                print(f"  Điểm đỉnh cao nhất     : {result['peak_score']} (Yêu cầu ngưỡng >= 8)")
                print("  Gợi ý khắc phục        :")
                print("    - Đảm bảo bài hát đang phát đã được thêm vào thư mục 'data/'.")
                print("    - Đặt loa điện thoại gần micro hơn để tín hiệu rõ ràng hơn.")
                print("    - Tránh tiếng người nói chuyện ồn ào xung quanh.")
                
            print("=" * 76)
            
        except Exception as err:
            print(f"\n[LỖI] Có lỗi xảy ra trong quá trình thu âm/nhận diện: {err}")

def main():
    parser = argparse.ArgumentParser(description="Live Shazam Microphone Demo (DSP Audio Fingerprinting)")
    parser.add_argument('--scan', type=str, default=DATA_DIR,
                        help="Thư mục chứa các bài hát (.mp3, .wav) để lập chỉ mục")
    parser.add_argument('--duration', type=float, default=5.0,
                        help="Thời gian thu âm mỗi lần (giây), mặc định: 5.0")
    args = parser.parse_args()
    
    run_live_recognition(scan_dir=args.scan, record_duration=args.duration)

if __name__ == '__main__':
    main()
