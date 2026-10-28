"""Play known WAVs on a PC speaker and extract the app's microphone recording.

Uses byte offsets in the running AV Tracker recording (whose WAV header is only
finalized when closed). Never plays through the Jabra itself: its echo canceller
could suppress the test. Output audio goes to build/asr-room-benchmark.
"""
import argparse
import json
import pathlib
import subprocess
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly, correlate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--adb', required=True)
parser.add_argument('--serial', default='42-00009ee8e085')
parser.add_argument('--device', type=int)
parser.add_argument('--case', default='reuniao')
parser.add_argument('--out', default='build/asr-room-benchmark')
args = parser.parse_args()
if args.device is None:
    args.device = next((i for i, device in enumerate(sd.query_devices())
                        if device['name'] == 'Speakers (Realtek(R) Audio)' and device['max_output_channels'] >= 2), None)
    if args.device is None:
        parser.error('Realtek speakers not found; select --device explicitly')
source = pathlib.Path('build/asr-benchmark')
folder = pathlib.Path(args.out)
folder.mkdir(parents=True, exist_ok=True)
remote = '/sdcard/Android/data/com.avtracker.mobile/files/realtime_sessions/.recording.wav'

def adb(*command):
    return subprocess.check_output([args.adb, '-s', args.serial, *command], text=True).strip()

manifest_path = folder / 'manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
for case in json.loads((source / 'manifest.json').read_text(encoding='utf-8')):
    if args.case != 'all' and case['id'] != args.case:
        continue
    audio, sr = sf.read(source / (case['id'] + '.wav'), dtype='float32')
    playback = resample_poly(audio, 441, 160)
    playback = np.concatenate([np.zeros(44100), playback, np.zeros(44100)]).astype('float32')
    start = int(adb('shell', 'stat', '-c', '%s', remote))
    print('Playing ' + case['id'], flush=True)
    sd.play(np.column_stack([playback, playback]), 44100, device=args.device, blocking=True)
    time.sleep(0.5)
    end = int(adb('shell', 'stat', '-c', '%s', remote))
    assert end > start, 'The app recording did not grow'
    snapshot = folder / 'recording-snapshot.tmp'
    adb('pull', remote, str(snapshot))
    with snapshot.open('rb') as recording:
        recording.seek(start)
        captured = np.frombuffer(recording.read(end-start), dtype='<i2').astype('float32') / 32768
    snapshot.unlink()
    sf.write(folder / (case['id'] + '.wav'), captured, 16000, subtype='PCM_16')
    corr = correlate(captured, audio, mode='valid', method='fft')
    offset = int(np.argmax(np.abs(corr)))
    aligned = captured[offset:offset+len(audio)]
    similarity = float(np.dot(aligned, audio)/(np.linalg.norm(aligned)*np.linalg.norm(audio)+1e-12))
    info = dict(case, durationSeconds=len(captured)/16000,
                capturedPeak=float(np.max(np.abs(captured))),
                capturedRms=float(np.sqrt(np.mean(captured**2))),
                waveformCorrelation=similarity, playbackDevice=str(sd.query_devices(args.device)['name']))
    manifest = [c for c in manifest if c['id'] != case['id']] + [info]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=True), flush=True)
