import sys
import numpy as np
import paddle
from matplotlib import pyplot as plt
from IPython import display as ipd
import soundfile as sf
import librosa.display
from parakeet.utils import display


from parakeet.models.lstm_speaker_encoder import LSTMSpeakerEncoder
print(help(LSTMSpeakerEncoder))

from examples.ge2e.audio_processor import SpeakerVerificationPreprocessor
from parakeet.models.lstm_speaker_encoder import LSTMSpeakerEncoder

# speaker encoder
p = SpeakerVerificationPreprocessor(
    sampling_rate=16000,
    audio_norm_target_dBFS=-30,
    vad_window_length=30,
    vad_moving_average_width=8,
    vad_max_silence_length=6,
    mel_window_length=25,
    mel_window_step=10,
    n_mels=40,
    partial_n_frames=160,
    min_pad_coverage=0.75,
    partial_overlap_ratio=0.5)
speaker_encoder = LSTMSpeakerEncoder(n_mels=40, num_layers=3, hidden_size=256, output_size=256)#语音特征的提取
speaker_encoder_params_path = "/home/aistudio/work/pretrained/ge2e_ckpt_0.3/step-3000000.pdparams"#pretrain 为预训练
speaker_encoder.set_state_dict(paddle.load(speaker_encoder_params_path))
speaker_encoder.eval()


# synthesizer
from parakeet.models.tacotron2 import Tacotron2
from examples.tacotron2_aishell3.chinese_g2p import convert_sentence
from examples.tacotron2_aishell3.aishell3 import voc_phones, voc_tones

from yacs.config import CfgNode        #目标音频特征的合成
synthesizer = Tacotron2(
    vocab_size=68,
    n_tones=10,
    d_mels= 80,
    d_encoder= 512,
    encoder_conv_layers = 3,
    encoder_kernel_size= 5,
    d_prenet= 256,
    d_attention_rnn= 1024,
    d_decoder_rnn = 1024,
    attention_filters = 32,
    attention_kernel_size = 31,
    d_attention= 128,
    d_postnet = 512,
    postnet_kernel_size = 5,
    postnet_conv_layers = 5,
    reduction_factor = 1,
    p_encoder_dropout = 0.5,
    p_prenet_dropout= 0.5,
    p_attention_dropout= 0.1,
    p_decoder_dropout= 0.1,
    p_postnet_dropout= 0.5,
    d_global_condition=256,
    use_stop_token=False
)
params_path = "/home/aistudio/work/pretrained/tacotron2_aishell3_ckpt_0.3/step-450000.pdparams"
synthesizer.set_state_dict(paddle.load(params_path))
synthesizer.eval()

# vocoder
from parakeet.models import ConditionalWaveFlow
vocoder = ConditionalWaveFlow(upsample_factors=[16, 16], n_flows=8, n_layers=8, n_group=16, channels=128, n_mels=80, kernel_size=[3, 3])
params_path = "/home/aistudio/work/pretrained/waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams"
vocoder.set_state_dict(paddle.load(params_path))
vocoder.eval()