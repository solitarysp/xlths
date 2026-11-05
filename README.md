# Mini-Shazam: Hệ thống Nhận diện Bài hát Số (DSP Audio Fingerprinting)
> **Đồ án / Bài tập lớn môn Xử lý Tín hiệu Số (XLTHS)**  
> **Sinh viên thực hiện:** Lê Văn Thành — **MSV:** K25DTCN402  
> **Nền tảng phát triển:** Python 3 (NumPy, SciPy, Matplotlib)

---

## 📌 Giới thiệu dự án

Dự án mô phỏng và hiện thực hóa nguyên lý hoạt động của ứng dụng nhận diện âm thanh nổi tiếng **Shazam** dựa hoàn toàn trên các nền tảng toán học của môn **Xử lý Tín hiệu Số (DSP)**:
* **Lấy mẫu & Rời rạc hóa:** $f_s = 22050 \text{ Hz}$ theo Định lý Nyquist-Shannon.
* **Biến đổi Fourier thời gian ngắn (STFT):** Phân tích tín hiệu âm nhạc không dừng trong miền thời gian - tần số.
* **Cửa sổ hóa Hanning:** Giảm thiểu hiện tượng rò rỉ phổ (Spectral Leakage).
* **Trích xuất đỉnh 2D (Constellation Map):** Bộ lọc cực đại cục bộ giảm chiều dữ liệu hơn 99%.
* **Ghép cặp Landmark Hashing:** Tạo khóa băm $[f_1 | f_2 | \Delta t]$ bất biến với thời gian và độ to.
* **Phân tích Histogram độ lệch thời gian:** Đối sánh và tìm đỉnh đồng thuận, nhận diện chính xác vị trí thời gian của đoạn trích ngay cả khi lẫn tạp âm trắng Gaussian (AWGN).

---

## 📁 Cấu trúc thư mục

```
xlths/
├── audio_generator.py          # Module tổng hợp bài hát mẫu bằng kỹ thuật DSP (Additive Synthesis, ADSR)
├── fingerprint.py              # Module cốt lõi DSP: STFT, Cửa sổ Hanning, Peak Extraction, Landmark Hashing
├── recognizer.py               # Module nhận diện: Bảng chỉ mục đảo (Inverted Index), Histogram Alignment
├── main_demo.py                # Script chạy Demo hoàn chỉnh (Kiểm thử 4 kịch bản thực tế)
├── generate_plots.py           # Script tự động xuất 5 biểu đồ chất lượng cao (300 DPI)
├── requirements.txt            # Danh sách thư viện cần thiết (numpy, scipy, matplotlib)
├── data/                       # Thư mục chứa các file âm thanh WAV mẫu đã tạo
├── output/                     # Thư mục lưu trữ biểu đồ xuất ra phục vụ báo cáo
├── BAO_CAO_BAI_TAP_LON_XLTHS.md # Báo cáo bài tập lớn môn Xử lý Tín hiệu Số hoàn chỉnh
└── README.md                   # Hướng dẫn sử dụng và giới thiệu dự án
```

---

## 🚀 Hướng dẫn cài đặt & Chạy chương trình

### 1. Cài đặt thư viện phụ thuộc
Hệ thống sử dụng các thư viện chuẩn trong Python cho DSP và `ffmpeg` (đã có sẵn trên máy) để đọc mọi định dạng âm thanh (.mp3, .wav, .m4a, .flac):
```bash
pip install -r requirements.txt
```

### 2. Thêm file nhạc MP3 của bạn vào hệ thống
Bạn có thể thêm bao nhiêu bài hát MP3 tùy thích theo 2 cách cực kỳ đơn giản:

* **Cách 1: Thả trực tiếp file vào thư mục `data/`**
  Chỉ cần copy các file `.mp3` (hoặc `.wav`, `.m4a`, `.flac`) vào thư mục `data/`. Khi chạy, chương trình sẽ tự động quét và nạp toàn bộ vào cơ sở dữ liệu.

* **Cách 2: Chỉ định một thư mục MP3 bất kỳ bên ngoài**
  Sử dụng tham số `--scan`:
  ```bash
  python3 main_demo.py --scan /path/to/thu_muc_mp3/
  ```

### 3. Các kịch bản chạy Demo và Nhận diện

#### A. 🎤 THU ÂM TRỰC TIẾP TỪ MICROPHONE (LIVE SHAZAM DEMO):
Bật bài hát trên điện thoại hoặc máy tính, sau đó chạy lệnh:
```bash
python3 live_mic_recognizer.py
# hoặc:
python3 main_demo.py --record
```
* Chương trình sẽ đếm ngược 3... 2... 1... và thu âm trực tiếp qua Micro máy tính (mặc định 5 giây).
* Tự động lọc đỉnh phổ FFT, trích xuất Landmark Hashes và trả về tên bài hát cùng vị trí giây phát thời gian thực!

#### B. Chạy tự động 4 kịch bản kiểm thử mặc định:
```bash
python3 main_demo.py
```

#### C. Nhận diện một đoạn nhạc MP3 bất kỳ:
```bash
python3 main_demo.py --query /path/to/doan_thu_am_3s.mp3
```

#### D. Thử nghiệm cắt một đoạn bất kỳ từ bài hát MP3 (kèm giả lập nhiễu ồn):
```bash
# Cắt 4 giây bắt đầu từ giây thứ 30 của bài MP3, thêm nhiễu trắng SNR = 10 dB:
python3 main_demo.py --slice-test /path/to/bai_hat.mp3 --start 30 --duration 4 --noise 10
```

### 4. Xuất lại các biểu đồ đồ án (`generate_plots.py`)
Lệnh chạy:
```bash
python3 generate_plots.py
```
Toàn bộ biểu đồ độ phân giải cao sẽ được lưu vào thư mục `output/`:
1. `plot_1_stft_and_peaks.png`: Dạng sóng thời gian, Phổ Spectrogram và Bản đồ chòm sao các đỉnh cực đại.
2. `plot_2_landmark_hashing.png`: Minh họa trực quan nguyên lý ghép cặp Điểm neo và Vùng mục tiêu (Target Zone).
3. `plot_3_noise_robustness.png`: So sánh phổ và bảo toàn đỉnh giữa tín hiệu sạch và tín hiệu dính nhiễu AWGN $\text{SNR} = 5 \text{ dB}$.
4. `plot_4_temporal_alignment_histogram.png`: Biểu đồ Histogram độ lệch thời gian giữa bài hát đúng và bài hát sai.
5. `plot_5_snr_accuracy_curve.png`: Đường cong đánh giá tỷ lệ nhận diện chính xác theo mức độ nhiễu SNR (từ -5 dB đến 30 dB).

---

## 📊 File Báo cáo Môn học
Chi tiết toàn bộ cơ sở lý thuyết toán học DSP, công thức đạo hàm, thiết kế thuật toán, bảng số liệu và phân tích kết quả nằm trong file:
👉 **[BAO_CAO_BAI_TAP_LON_XLTHS.md](BAO_CAO_BAI_TAP_LON_XLTHS.md)**
*(Có thể mở bằng Visual Studio Code, Typora hoặc copy sang Microsoft Word để nộp cho Giảng viên).*
