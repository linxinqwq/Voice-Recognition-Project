import sys
import webrtcvad
import collections
import signal
import pyaudio
from array import array
from struct import pack
import wave
import time

# 音频录制参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30  # 支持10、20和30毫秒
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # 读取的块大小
CHUNK_BYTES = CHUNK_SIZE * 2  # 16位PCM = 2字节
PADDING_DURATION_MS = 1500  # 1.5秒填充时长
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400毫秒窗口大小
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

# 初始化WebRTC VAD
vad = webrtcvad.Vad(1)

# 初始化PyAudio
pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 start=False,
                 frames_per_buffer=CHUNK_SIZE)

got_a_sentence = False
leave = False


def handle_int(sig, frame):
    """处理中断信号以停止录音。"""
    global leave, got_a_sentence
    leave = True
    got_a_sentence = True


def record_to_file(path, data, sample_width):
    """将录制的数据保存到WAV文件。"""
    data = pack('<' + ('h' * len(data)), *data)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data)


def normalize(snd_data):
    """标准化录制数据的音量。"""
    MAXIMUM = 32767
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    return array('h', [int(i * times) for i in snd_data])


signal.signal(signal.SIGINT, handle_int)

while not leave:
    ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False
    raw_data = array('h')
    ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
    ring_buffer_index = 0
    ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
    ring_buffer_index_end = 0
    index = 0
    start_point = 0
    start_time = time.time()
    print("* 录音中:")
    stream.start_stream()

    while not got_a_sentence and not leave:
        chunk = stream.read(CHUNK_SIZE)
        raw_data.extend(array('h', chunk))
        index += CHUNK_SIZE
        time_use = time.time() - start_time

        active = vad.is_speech(chunk, RATE)
        sys.stdout.write('1' if active else '_')
        ring_buffer_flags[ring_buffer_index] = 1 if active else 0
        ring_buffer_index = (ring_buffer_index + 1) % NUM_WINDOW_CHUNKS

        ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
        ring_buffer_index_end = (ring_buffer_index_end + 1) % NUM_WINDOW_CHUNKS_END

        if not triggered:
            ring_buffer.append(chunk)
            if sum(ring_buffer_flags) > 0.8 * NUM_WINDOW_CHUNKS:
                sys.stdout.write(' 开始 ')
                triggered = True
                start_point = index - CHUNK_SIZE * 20
                ring_buffer.clear()
        else:
            ring_buffer.append(chunk)
            if (NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end) > 0.90 * NUM_WINDOW_CHUNKS_END or
                    time_use > 10):
                sys.stdout.write(' 结束 ')
                triggered = False
                got_a_sentence = True

        sys.stdout.flush()

    sys.stdout.write('\n')
    stream.stop_stream()
    print("* 录音结束")
    got_a_sentence = False

    # 写入文件
    raw_data = normalize(raw_data[start_point:])
    record_to_file("vadoutput.wav", raw_data, 2)
    leave = True

stream.close()
pa.terminate()
