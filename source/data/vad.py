import collections
import contextlib
import sys
import wave
import webrtcvad

def _read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def _frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    triggered = False
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    cnt = 0
    start = -1
    voiced_frames = []
    final_start = 1e4
    final_end = -1

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        cnt += 1
        if not triggered:
            ring_buffer.append(is_speech)
            num_voiced = len([speech for speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for s in ring_buffer:
                    voiced_frames.append(s)
                    start = max(cnt - ring_buffer.maxlen, 0)
        else:
            voiced_frames.append(is_speech)
            ring_buffer.append(is_speech)
            num_unvoiced = len([speech for speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                end = cnt
                while not voiced_frames[0]:
                    start += 1
                    del voiced_frames[0]
                while not voiced_frames[-1]:
                    end -= 1
                    del voiced_frames[-1]
                ring_buffer.clear()
                final_start = min(final_start, start)
                final_end = max(final_end, end)
                voiced_frames = []

    if triggered:
        end = cnt
        while not voiced_frames[0]:
            start += 1
            del voiced_frames[0]
        while not voiced_frames[-1]:
            end -= 1
            del voiced_frames[-1]
        final_start = min(final_start, start)
        final_end = max(final_end, end)

    return final_start * frame_duration_ms, final_end * frame_duration_ms

def get_interval(wav_dir):
    audio, sample_rate = _read_wave(wav_dir)
    vad = webrtcvad.Vad(2)
    frames = _frame_generator(10, audio, sample_rate)
    frames = list(frames)
    start, end = vad_collector(sample_rate, 10, 200, vad, frames)
    return start, end
