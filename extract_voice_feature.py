import initialize_model
#import initialize_model
import paddle
import os
import numpy as np
from initialize_model import p,speaker_encoder,synthesizer,vocoder
from examples.tacotron2_aishell3.chinese_g2p import convert_sentence
from examples.tacotron2_aishell3.aishell3 import voc_phones, voc_tones
import soundfile as sf
ref_name = "youger_04.wav"
ref_audio_path = f"./ref_audio/{ref_name}"
#ipd.Audio(filename=ref_audio_path)
mel_sequences = p.extract_mel_partials(p.preprocess_wav(ref_audio_path))
print("mel_sequences: ", mel_sequences.shape)
with paddle.no_grad():
    embed = speaker_encoder.embed_utterance(paddle.to_tensor(mel_sequences))
print("embed shape: ", embed.shape)


sentence = "祝%各位飞桨%开发者们$七夕%情人节%快乐$"

phones, tones = convert_sentence(sentence)
print(phones)
print(tones)

phones = np.array([voc_phones.lookup(item) for item in phones], dtype=np.int64)
tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)

phones = paddle.to_tensor(phones).unsqueeze(0)
tones = paddle.to_tensor(tones).unsqueeze(0)
utterance_embeds = paddle.unsqueeze(embed, 0)

with paddle.no_grad():
    outputs = synthesizer.infer(phones, tones=tones, global_condition=utterance_embeds)
mel_input = paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1])

#fig = display.plot_alignment(outputs["alignments"][0].numpy().T)

with paddle.no_grad():
    wav = vocoder.infer(mel_input)
wav = wav.numpy()[0]
if not os.path.exists("./data/syn_audio/"):
    os.makedirs("./data/syn_audio/")
sf.write(f"./data/syn_audio/{ref_name}", wav, samplerate=22050)
#librosa.display.waveplot(wav)