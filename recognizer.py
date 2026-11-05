"""
recognizer.py
Mô-đun Quản lý Cơ sở Dữ liệu Fingerprint & Thuật toán Khớp / Nhận diện bài hát:
1. Chỉ mục đảo (Inverted Index / Hash Table) lưu trữ dấu vân tay các bài hát.
2. Thuật toán phân tích biểu đồ sai lệch thời gian (Time-Offset Histogram Analysis).
3. Đánh giá độ tin cậy (Confidence Score) và phát hiện bài hát ngoại lai (Unknown Song).
"""

from collections import defaultdict, Counter
import numpy as np
from fingerprint import fingerprint_audio, HOP_LENGTH, DEFAULT_FS

class MiniShazamRecognizer:
    def __init__(self, min_peak_score: int = 8, min_confidence: float = 0.20):
        """
        Khởi tạo hệ thống nhận diện bài hát:
        - db: Bảng băm chỉ mục đảo (hash_key -> list[(song_id, song_time_offset)])
        - songs: Từ điển thông tin bài hát (song_id -> metadata)
        - min_peak_score: Ngưỡng số lượng cặp khớp tối thiểu tại cùng một độ lệch thời gian
        - min_confidence: Ngưỡng tỉ lệ tin cậy tối thiểu (peak_score / total_matches)
        """
        self.db = defaultdict(list)
        self.songs = {}
        self.next_song_id = 1
        self.min_peak_score = min_peak_score
        self.min_confidence = min_confidence
        
    def add_song(self, title: str, audio: np.ndarray) -> int:
        """
        Trích xuất đặc trưng và nạp bài hát vào cơ sở dữ liệu.
        """
        song_id = self.next_song_id
        self.next_song_id += 1
        
        duration_sec = len(audio) / DEFAULT_FS
        self.songs[song_id] = {
            'song_id': song_id,
            'title': title,
            'duration_sec': duration_sec,
            'total_samples': len(audio)
        }
        
        # Trích xuất toàn bộ landmark hashes của bài hát
        hashes, peaks, spec_db = fingerprint_audio(audio, song_id=song_id)
        
        # Lưu vào bảng băm chỉ mục đảo
        for hash_key, t_offset, _ in hashes:
            self.db[hash_key].append((song_id, t_offset))
            
        self.songs[song_id]['num_peaks'] = len(peaks)
        self.songs[song_id]['num_hashes'] = len(hashes)
        
        return song_id

    def recognize(self, query_audio: np.ndarray) -> dict:
        """
        Nhận diện một đoạn âm thanh trích đoạn (Query Snippet).
        
        Nguyên lý DSP & Đối sánh thời gian:
        1. Trích xuất dấu vân tay của đoạn query: {(hash_key, t_query)}
        2. Với mỗi hash_key, tìm kiếm trong DB các bản ghi (song_id, t_song).
        3. Tính độ lệch thời gian: delta_t = t_song - t_query.
           Nếu đoạn query thực sự thuộc về bài hát song_A tại thời điểm t_start,
           thì hầu hết các điểm khớp đúng đều có delta_t xấp xỉ t_start (tạo thành một đỉnh nhọn cực đại
           trong biểu đồ Histogram của delta_t).
        4. Đối với các bài hát khác, các điểm khớp ngẫu nhiên (nhiễu/trùng lặp) sẽ phân bố rải rác.
        """
        query_hashes, query_peaks, query_spec = fingerprint_audio(query_audio)
        
        if not query_hashes:
            return {
                'match': False,
                'song_id': None,
                'title': 'Không phát hiện đủ đặc trưng (Tín hiệu quá yếu/ngắn)',
                'confidence': 0.0,
                'peak_score': 0,
                'offset_sec': 0.0,
                'query_hashes_count': 0,
                'all_scores': {},
                'best_histogram': {}
            }
            
        # Gom nhóm các độ lệch thời gian theo từng song_id:
        # matches_per_song[song_id] = list of delta_t
        matches_per_song = defaultdict(list)
        
        for hash_key, t_query, _ in query_hashes:
            if hash_key in self.db:
                for song_id, t_song in self.db[hash_key]:
                    delta_t = t_song - t_query
                    matches_per_song[song_id].append(delta_t)
                    
        if not matches_per_song:
            return {
                'match': False,
                'song_id': None,
                'title': 'Không tìm thấy bài hát phù hợp (Chưa có trong CSDL)',
                'confidence': 0.0,
                'peak_score': 0,
                'offset_sec': 0.0,
                'query_hashes_count': len(query_hashes),
                'all_scores': {},
                'best_histogram': {}
            }
            
        best_song_id = None
        highest_peak_score = 0
        best_offset_frame = 0
        best_confidence = 0.0
        best_histogram_data = {}
        all_song_scores = {}
        
        # Đánh giá Histogram cho từng bài hát có xuất hiện trong kết quả tra cứu
        for song_id, delta_list in matches_per_song.items():
            # Đếm tần suất xuất hiện của từng giá trị delta_t
            counts = Counter(delta_list)
            peak_offset, peak_count = counts.most_common(1)[0]
            total_song_matches = len(delta_list)
            
            # Độ tin cậy = số lượng cặp khớp tại đỉnh / tổng số cặp khớp tìm thấy với bài hát đó
            confidence = peak_count / total_song_matches if total_song_matches > 0 else 0.0
            
            all_song_scores[song_id] = {
                'title': self.songs[song_id]['title'],
                'peak_score': peak_count,
                'total_matches': total_song_matches,
                'confidence': confidence,
                'offset_sec': (peak_offset * HOP_LENGTH) / DEFAULT_FS
            }
            
            if peak_count > highest_peak_score:
                highest_peak_score = peak_count
                best_song_id = song_id
                best_offset_frame = peak_offset
                best_confidence = confidence
                best_histogram_data = dict(counts)
                
        # Kiểm tra điều kiện ngưỡng chấp nhận (Threshold Check)
        is_recognized = (
            highest_peak_score >= self.min_peak_score and
            best_confidence >= self.min_confidence
        )
        
        best_title = self.songs[best_song_id]['title'] if is_recognized else 'Không nhận diện được (Dưới ngưỡng tin cậy)'
        best_offset_sec = (best_offset_frame * HOP_LENGTH) / DEFAULT_FS if is_recognized else 0.0
        
        return {
            'match': is_recognized,
            'song_id': best_song_id if is_recognized else None,
            'title': best_title,
            'confidence': best_confidence if is_recognized else 0.0,
            'peak_score': highest_peak_score,
            'offset_sec': best_offset_sec,
            'query_hashes_count': len(query_hashes),
            'all_scores': all_song_scores,
            'best_histogram': best_histogram_data
        }
