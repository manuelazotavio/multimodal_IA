"""Render the results from AzureSpeechBenchmarkTest without additional dependencies."""
import json
import re
import unicodedata
from pathlib import Path

def words(text):
    # Explicit equivalences present in this corpus; preserve all genuine word/name errors.
    text = text.lower().replace('15h30', 'quinze horas e trinta minutos')
    text = re.sub(r'\b15\b', 'quinze', text)
    text = re.sub(r'\b30\b', 'trinta', text)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))
    return re.findall('[a-z0-9]+', text)

def distance(a, b):
    row = list(range(len(b) + 1))
    for i, left in enumerate(a):
        next_row = [i + 1]
        for j, right in enumerate(b):
            next_row.append(min(next_row[j] + 1, row[j+1] + 1, row[j] + (left != right)))
        row = next_row
    return row[-1]

folder = Path('build/asr-benchmark')
results = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
lines = ['# Avaliação de transcrição com Azure Speech', '',
         'Data: 27/10/2026. Quatro frases sintéticas em português, duas vozes Azure (Francisca e Antonio), uma frase a +20% de velocidade. PCM mono, 16 kHz, 16 bits.', '',
         'Motor Kotlin/ONNX do AV Tracker, com Silero VAD e as configurações de produção de cada modelo. Execução no PC: os tempos abaixo não medem a velocidade na Jabra. Não inclui microfone, acústica, diarização ou recortes de fala da sessão ao vivo.', '',
         'WER = (substituições + exclusões + inserções) / palavras de referência. Ignora pontuação, acentos e maiúsculas; hífens separam palavras. Não normaliza números escritos por extenso para algarismos. Amostra pequena: os resultados não estimam a precisão de uma reunião.', '',
         'Também mostramos WER com equivalência explícita de horários: “15h30”, “15 horas e 30 minutos” e “quinze horas e trinta minutos”. Isso evita contar uma representação correta de horário como falha de reconhecimento. Os erros em nomes permanecem.', '',
         '| Modelo | Erros literais / palavras | WER literal | WER com horário equivalente | Processamento total no PC |',
         '|---|---:|---:|---:|---:|']
for model in ['base', 'small']:
    rows = [r for r in results if r['model'] == model]
    assert len(rows) == len(manifest), f'Incomplete results for {model}'
    errors = sum(r['wordErrors'] for r in rows)
    total_words = sum(r['referenceWords'] for r in rows)
    equivalent_errors = sum(distance(words(r['reference']), words(r['transcription'])) for r in rows)
    seconds = sum(r['processingSeconds'] for r in rows)
    lines.append(f'| {model} | {errors} / {total_words} | {100*errors/total_words:.2f}% | {100*equivalent_errors/total_words:.2f}% ({equivalent_errors}/{total_words}) | {seconds:.2f} s |')
lines += ['', '## Comparações por áudio', '']
for case in manifest:
    lines += [f"### {case['id']}", '', f"Voz: {case['voice']}; velocidade: {case['rate']}; duração: {case['durationSeconds']:.2f} s.", '',
              f"[Ouvir áudio]({case['id']}.wav)", '', f"**Esperado:** {case['text']}", '']
    for row in (r for r in results if r['id'] == case['id']):
        lines += [f"**{row['model']}:** {row['transcription']}", '',
                  f"Erros: {row['wordErrors']}/{row['referenceWords']} ({100*row['wer']:.2f}%); tempo: {row['processingSeconds']:.2f} s.", '']
(folder / 'relatorio.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:17]))
