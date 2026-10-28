# AV Tracker e VocalID na Jabra

Na tela inicial, marque **Usar VocalID para vozes e transcrição** para delegar o áudio.
Por padrão, o AV Tracker usa seu próprio áudio e sua própria transcrição.
Nesse modo, o AV Tracker carrega somente YOLO e EdgeFace, mantém os perfis faciais
existentes e não abre o microfone nem inicia os modelos locais de áudio. O botão
**Abrir VocalID** abre o aplicativo `com.vocalid.jabra`.

Inicie a reunião na tela do VocalID e retorne ao AV Tracker. O serviço de áudio
do VocalID continua em primeiro plano enquanto o AV Tracker mostra as faces.
Os falantes e a transcrição permanecem na tela e no histórico do VocalID.
O botão Encerrar sessão do AV Tracker encerra somente a sessão facial; a sessão
de áudio deve ser encerrada no próprio VocalID.

Desmarcar a opção restaura o processamento multimodal local anterior. Seus modelos,
perfis, transcrição e identificação de voz continuam no projeto e no APK.

## Limite da integração atual

O VocalID Jabra instalado, versão `0.2.0-jabra.1`, expõe sua atividade principal,
mas o serviço `com.vocalid.android.audio.RecordingService` não é exportado.
Não foi encontrada uma API pública de resultados no manifesto do APK.
Assim, este modo coordena o uso dos dispositivos, mas não importa automaticamente
nomes ou transcrições do VocalID e não associa suas vozes aos rostos.
Para isso, o responsável pelo VocalID precisa disponibilizar uma API/SDK ou uma
versão que compartilhe eventos documentados de falante e transcrição com timestamps.

## Transcrição

O APK do VocalID contém Whisper Base multilíngue INT8, fixado em assets:
`models/base-encoder.int8.onnx`, `models/base-decoder.int8.onnx` e `models/base-tokens.txt`.
O carregador usa diretamente o AssetManager e configura português (`pt`). Não foi
encontrada seleção de modelo externo nessa implementação.

O Whisper Small multilíngue INT8 é um candidato para comparar: há arquivos ONNX
no formato Sherpa, com aproximadamente 375 MB para encoder, decoder e tokens.
Não usar a variante `small.en`, que aceita somente inglês. O ganho de qualidade e
o tempo de transcrição precisam ser medidos com a mesma gravação em português e
na Jabra; um modelo maior não garante uma melhora na captura ou na segmentação.
Os modelos Sherpa não são intercambiáveis com os grafos personalizados produzidos
por `scripts/export_whisper.py` para o modo multimodal legado do AV Tracker.

Fontes: [Whisper](https://github.com/openai/whisper),
[exportações oficiais do Sherpa](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html),
[Whisper Small no formato Sherpa](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small).
