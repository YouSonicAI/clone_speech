import paddle
import numpy as np
from initialize_model import p,speaker_encoder,synthesizer,vocoder
from examples.tacotron2_aishell3.chinese_g2p import convert_sentence
from examples.tacotron2_aishell3.aishell3 import voc_phones, voc_tones
import soundfile as sf
import os
lable = 1  # 根据上面的选择器写入相应的值
sentences = "虎起生活的风帆，走向虎关通途。"  # 需要写入的祝福语
photo_patch = "./靓照.jpg"  # 照片地址
custom = "./" # 自定义语音地址

tone_gather = {1:'./ref_audio/台湾腔小姐姐.wav',
2:'./ref_audio/小姐姐.wav',
3:'./ref_audio/蜡笔小新.wav',
4:'./ref_audio/东北老铁.wav',
5:'./ref_audio/粤语小哥哥.wav',
6:'./ref_audio/小哥哥.wav',
7:'./ref_audio/低沉大叔.wav',
8:'./ref_audio/小宝宝.wav',
9:'./ref_audio/御姐.wav',
10:'./ref_audio/萝莉.wav'}

tone_gather[11] = custom

if (custom == "./" and lable == 11) or (lable not in [i for i in range(1,12)]):
    lable = 1


symbol = [',', '.', '，', '。','!', '！', ';', '；', ':', "："]
sentence = ''
for i in sentences:
    if i in symbol:
        sentence = sentence[:-1] + '$'
    else:
        sentence = sentence + i + '%'

label_list = [1,2,3,4,5,6,7,8,9,10]
for label in label_list:
    ref_audio_path = tone_gather[lable]
    mel_sequences = p.extract_mel_partials(p.preprocess_wav(ref_audio_path))
    # print("mel_sequences: ", mel_sequences.shape)
    with paddle.no_grad():
        embed = speaker_encoder.embed_utterance(paddle.to_tensor(mel_sequences))
    # print("embed shape: ", embed.shape)
    phones, tones = convert_sentence(sentence)
    # print(phones)
    # print(tones)

    phones = np.array([voc_phones.lookup(item) for item in phones], dtype=np.int64)
    tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)

    phones = paddle.to_tensor(phones).unsqueeze(0)
    tones = paddle.to_tensor(tones).unsqueeze(0)
    utterance_embeds = paddle.unsqueeze(embed, 0)
    with paddle.no_grad():
        outputs = synthesizer.infer(phones, tones=tones, global_condition=utterance_embeds)
    mel_input = paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1])
    #fig = display.plot_alignment(outputs["alignments"][0].numpy().T)
    #os.system('mkdir -p /home/aistudio/syn_audio/')
    with paddle.no_grad():
        wav = vocoder.infer(mel_input)
    wav = wav.numpy()[0]
    synthesize_dir ="./data/syn_audio/"
    if not os.path.exists(synthesize_dir):
        os.makedirs(synthesize_dir)
    file_name = ref_audio_path.split("/")[-1]
    sf.write(os.path.join(synthesize_dir,file_name), wav, samplerate=22050)
