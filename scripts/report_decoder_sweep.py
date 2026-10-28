"""Summarize the local decoder experiment, without playing audio."""
import json
import re
import unicodedata
from pathlib import Path

folder = Path('build/asr-decoder-sweep')
results = json.loads((folder / 'results.json').read_text(encoding='utf-8'))

def words(text):
    text = text.lower().replace('15h30', 'quinze horas e trinta minutos')
    text = re.sub(r'\b15\b', 'quinze', text)
    text = re.sub(r'\b30\b', 'trinta', text)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))
    return re.findall('[a-z0-9]+', text)

def distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left):
        next_row = [i + 1]
        for j, b in enumerate(right):
            next_row.append(min(next_row[j] + 1, row[j+1] + 1, row[j] + (a != b)))
        row = next_row
    return row[-1]

lines = ['# Experimento com opções do Whisper Small', '',
         'Os mesmos cinco arquivos: quatro WAVs Azure e a gravação acústica já capturada pela Jabra. Nenhuma nova reprodução. O texto esperado só é usado na avaliação; não é fornecido ao transcritor.', '',
         'Todas as opções usam o VAD como gate, preservando o áudio quando há fala. Production é a configuração anterior ao experimento: beam 3 e prompt genérico de português. No-prompt: beam 3 sem o prompt. Beam1: uma hipótese e o prompt genérico. No-prompt-beam5: cinco hipóteses, sem o prompt.', '',
         'Erros calculados por distância de edição entre palavras, ignorando acentos, caixa e pontuação. Os horários 15h30 e quinze horas e trinta minutos são equivalentes. Tempos no PC, não na Jabra. Amostra pequena, sem garantia de generalização.', '',
         '| Configuração | Erros: áudio limpo / 71 palavras | Erros: microfone / 17 palavras | Tempo total no PC |',
         '|---|---:|---:|---:|']
for variant in ['small-production', 'small-no-prompt', 'small-beam1', 'small-no-prompt-beam5']:
    rows = [r for r in results if r['model'] == variant]
    assert len(rows) == 5, f'Incomplete results for {variant}'
    clean = sum(distance(words(r['reference']), words(r['transcription'])) for r in rows if not r['id'].startswith('mic-'))
    mic = sum(distance(words(r['reference']), words(r['transcription'])) for r in rows if r['id'].startswith('mic-'))
    seconds = sum(r['processingSeconds'] for r in rows)
    lines.append(f'| {variant} | {clean}/71 | {mic}/17 | {seconds:.2f} s |')
lines += ['', '## Saídas completas', '']
for row in results:
    lines += [f"### {row['model']} / {row['id']}", '', '**Esperado:** '+row['reference'], '', '**Saída:** '+row['transcription'], '']
(folder / 'relatorio.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:13]))
