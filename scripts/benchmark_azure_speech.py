"""Generate a small Portuguese corpus using the Duca project's Azure resource.

Credentials are read locally, never copied into the benchmark output.
Run the opt-in AzureSpeechBenchmarkTest afterwards to compare Base and Small.
"""
import argparse
import json
import pathlib
import wave
from xml.sax.saxutils import escape

import azure.cognitiveservices.speech as speechsdk

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--key-file', required=True)
parser.add_argument('--endpoint', default='https://ii-ia.cognitiveservices.azure.com/')
parser.add_argument('--out', default='build/asr-benchmark')
args = parser.parse_args()
folder = pathlib.Path(args.out)
folder.mkdir(parents=True, exist_ok=True)
cases = [
    ('reuniao', 'pt-BR-FranciscaNeural', '0%',
     'Bom dia a todos. Nesta reunião vamos revisar os resultados do aplicativo e definir os próximos passos.'),
    ('camera', 'pt-BR-AntonioNeural', '0%',
     'A câmera precisa identificar os rostos enquanto o microfone captura a fala de cada participante.'),
    ('nomes-numeros', 'pt-BR-FranciscaNeural', '0%',
     'Manuela e João vão apresentar o relatório de Ilhabela na terça-feira, às quinze horas e trinta minutos.'),
    ('fala-rapida', 'pt-BR-AntonioNeural', '+20%',
     'Eu acho que podemos começar pelo teste de áudio, mas precisamos conferir se a transcrição acompanha a conversa sem perder palavras.'),
]
key = pathlib.Path(args.key_file).read_text(encoding='utf-8-sig').strip()
config = speechsdk.SpeechConfig(subscription=key, endpoint=args.endpoint)
config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
synth = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
manifest = []
for ident, voice, rate, text in cases:
    path = folder / (ident + '.wav')
    if not path.exists():
        ssml = ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">'
                f'<voice name="{voice}"><prosody rate="{rate}">{escape(text)}</prosody></voice></speak>')
        result = synth.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError('Azure synthesis failed: ' + str(result.cancellation_details.reason))
        path.write_bytes(result.audio_data)
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        duration = wav.getnframes() / wav.getframerate()
    manifest.append(dict(id=ident, voice=voice, rate=rate, text=text, durationSeconds=duration))
    print(f'{ident}: {duration:.2f}s', flush=True)
(folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
