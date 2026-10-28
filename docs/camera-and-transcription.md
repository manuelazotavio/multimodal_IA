# Alinhamento da câmera e transcrição local

## Câmera

Na PanaCast 40, a combinação anterior de análise 640x480 com prévia de outra
resolução produzia enquadramentos diferentes, incluindo faixas pretas no buffer
analisado. Escalar as caixas diretamente pela largura/altura da View deslocava o rosto.

Preview e ImageAnalysis agora pedem 1280x720, usam o mesmo ViewPort/UseCaseGroup e
o OverlayView converte as coordenadas com ImageProxyTransformFactory e
CoordinateTransform, incluindo recorte e rotação. A prévia usa TextureView
(PreviewView.COMPATIBLE). Não são aplicados deslocamentos fixos específicos da sala.

Os bitmaps temporários do analisador são reciclados depois da inferência. O horário
do quadro é medido antes do processamento para evitar somar o tempo da inferência à
sincronização com o áudio. A atualização da detecção continua limitada pelo tempo
de inferência; movimento rápido pode apresentar atraso em relação à prévia ao vivo.

Referência: https://developer.android.com/media/camera/camerax/transform-output

## Transcrição

O AV Tracker pode usar Whisper Base (`assets/whisper/`) ou Small multilíngue
(`assets/whisper-small/`), ambos executados localmente pelo mesmo motor Kotlin/ONNX.
Small é oferecido somente quando seus quatro arquivos estão presentes e fica
selecionado inicialmente nesse caso. Base continua disponível como alternativa mais leve.
O modelo escolhido é salvo nas preferências e registrado nas métricas da sessão.

Os arquivos do Small são gerados por:

```powershell
python scripts/export_whisper.py --model-dir <pasta-do-openai-whisper-small> --out src/main/assets/whisper-small
```

Eles usam armazenamento FP16 e computação FP32. Os grafos Sherpa-ONNX baixados para
pesquisar o VocalID têm outro contrato de entrada/saída e não substituem esses arquivos.
Os diretórios gerados não são versionados no Git.

Small usa beam search 3 e temperatura 0 para evitar múltiplas tentativas de amostragem
do mesmo áudio em um dispositivo limitado. Base mantém as opções anteriores.
O teste WhisperSmallTest cobre português conhecido tanto em decodificação greedy
quanto no fluxo com VAD e as opções usadas no aplicativo. Isso verifica o funcionamento
do modelo, mas não constitui uma medição de precisão em reuniões ou ambientes ruidosos.

## Diagnóstico da captação (27/10/2026)

O Android informou entrada MIC interna, PCM16 mono a 16 kHz, não silenciada,
sem efeitos de software registrados no AudioFlinger. A configuração do fabricante
usa hardware a 48 kHz; a interface de entrada aceita 16 kHz. Não houve clipping
na amostra. Isso não exclui processamento no DSP do fabricante ou problemas acústicos.

Após a investigação ao vivo, a captura passou a solicitar `VOICE_RECOGNITION`
(confirmado pelo Android como source 6), com fallback para MIC caso o preset não
inicialize. A troca de preset não comprovou melhora na frase captada: não deve
ser interpretada como uma solução validada para a acústica da sala. Os logs agora
registram os motivos de rejeição e as métricas, sem o conteúdo descartado.

Na frase Azure reproduzida pelo PC e captada pelo app, o VAD descartou intervalos
que continham palavras. Small teve 11 erros em 17 palavras com a concatenação dos
trechos aprovados pelo VAD; sem essa remoção, recuperou a frase com duas substituições
(“definir os” por “depender dos”). O alinhamento com o original foi verificado por correlação
da forma de onda, sem reproduzir novamente o áudio.

As opções de produção de Base e Small agora usam `SPEECH_GATE`: o VAD ainda
rejeita clipes sem fala detectada, mas preserva o áudio inteiro quando encontra fala.
Isso evita destruir palavras que o VAD classificou incorretamente como silêncio.
O modo `REMOVE_SILENCE` continua disponível como padrão da API para os testes de
paridade com faster-whisper. Há regressão opt-in com a captura local em
`build/asr-room-benchmark`, incluindo rejeição de silêncio.

Essa avaliação não mede todos os cortes feitos pela diarização ao vivo. O ganho
observado é desta gravação conhecida, não uma estimativa de precisão de reuniões.

## Opções de decodificação

Uma comparação local de cinco arquivos (quatro originais Azure e uma captura da
Jabra), sem nova reprodução, não encontrou redução de erros ao remover o prompt
genérico ou alterar beam 3 para beam 1 ou 5. Beam 3 sem o prompt genérico preservou
as transcrições, exceto pontuação, e consumiu menos tempo no PC. Small agora usa
essa configuração; português continua selecionado pelo token de idioma. Base não
teve suas opções de prompt alteradas. O resultado não estabelece o ganho de tempo
na Jabra nem garante a mesma precisão em outras conversas.

## Modo Base para menor latência

Os logs da Jabra mostraram Small consumindo 15–22 s por trecho de aproximadamente
2 s, mesmo depois de remover o prompt. Para priorizar a conversa ao vivo, Base usa
beam 1, temperatura 0 sem novas tentativas e sem prompt genérico, mantendo o gate
de fala. A amostra acústica passou de 2 erros no Small para 7 no Base rápido:
há uma troca concreta entre precisão e velocidade.

Quando Base está selecionado e a diarização completa está desligada, o agendador
consome 4 s novos por janela, com até 4 s anteriores de contexto. A primeira janela
sai aos 4 s; as seguintes aos 8, 12, 16 s, sem a espera anterior de 15 s pela segunda
janela. As zonas novas são contíguas e não se sobrepõem. O agendamento original
continua disponível no modo Small/diarização completa. Essa cadência não inclui
o tempo de inferência: a interface continua exibindo resultados por segmentos,
sem hipóteses parciais palavra por palavra.

O Tiny multilíngue também foi exportado e avaliado, mas rejeitado para a captura
real: 16 erros em 17 palavras. Os arquivos de pesquisa ficam fora dos assets do
APK, em `.codex/tools/models/whisper-tiny-onnx` no PC. O runtime permanece na Jabra.

No Android, as sessões encoder/decoder do Whisper agora usam duas threads por
operador, uma thread entre operadores e não fazem busy-wait quando o pool está
ocioso. As sessões desktop de referência continuam com os padrões anteriores.
Essa é uma tentativa de reduzir disputa entre sessões de visão e áudio; seu ganho
precisa ser medido no aparelho, não inferido dos testes no PC.
Referência: https://onnxruntime.ai/docs/performance/tune-performance/threading.html

## Proteções contra repetição e geração excessiva

O modo de produção limita a geração por janela a `min(224, max(32,
ceil(segundos * 12 + 16)))` tokens. Se o limite for atingido sem fim de texto,
descarta o resultado em vez de exibir uma frase cortada ou uma repetição sem fim.
Também descarta tentativas com compressão excessiva (acima de 2,4), log-probabilidade
média abaixo de -2,0, ou abaixo de -1,0 junto de probabilidade de ausência de fala
acima de 0,3. Uma nota abaixo de -1,0 sem esse indicativo de ausência de fala pode
ser exibida como “Texto incerto”, mantendo o marcador no arquivo salvo. Esses textos
não alimentam extração de nomes nem a análise por LLM. Essas proteções são opt-in na API e estão ativadas para Base
e Small no aplicativo; os modos de referência mantêm seu comportamento anterior.

O filtro de exibição rejeita sequências consecutivas de 2–6 palavras repetidas
quatro vezes, mesmo com uma introdução diferente e sem pontuação. Isso cobre o
caso observado de “e depois ... de a parte de a parte ...”, que escapava do filtro
antigo de frases inteiramente idênticas. A proteção pode omitir fala legítima muito
repetitiva ou de baixa confiança; não impede todos os erros fluentes de reconhecimento.
