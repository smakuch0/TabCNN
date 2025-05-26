import numpy as np
import librosa
from TabCNN import TabCNN

def preprocess_audio(audio_file, spec_repr="c"):
    y, sr = librosa.load(audio_file, sr=22050)
    y = librosa.util.normalize(y)
    
    hop_length = 512
    if spec_repr == "c":
        data = np.abs(librosa.cqt(y, hop_length=hop_length, sr=sr, n_bins=192, bins_per_octave=24))
    elif spec_repr == "m":
        data = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_length)
    elif spec_repr == "s":
        data = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    
    return np.swapaxes(data, 0, 1)

def predict_custom_audio(audio_file, weights_path):
    print(" === loading model... === ")
    tabcnn = TabCNN()
    tabcnn.build_model()
    tabcnn.model.load_weights(weights_path)
    
    print(" === preprocessing audio... === ")
    audio_repr = preprocess_audio(audio_file)
    con_win_size = 9
    halfwin = con_win_size // 2

    print(" === predicting... === ")
    
    predictions = []
    padded_repr = np.pad(audio_repr, [(halfwin, halfwin), (0, 0)], mode='constant')
    
    for frame_idx in range(len(audio_repr)):
        window = padded_repr[frame_idx:frame_idx + con_win_size]
        X = np.expand_dims(np.expand_dims(np.swapaxes(window, 0, 1), -1), 0)
        pred = tabcnn.model.predict(X, verbose=0)
        predictions.append(pred[0])
    
    return np.array(predictions)

def print_tab(tab_data, start=0, length=50):
    strings = ['E', 'A', 'D', 'G', 'B', 'E']
    tab_section = tab_data[start:start+length]
    lines = []
    
    for i in range(5, -1, -1):
        line = strings[i] + '|'
        for frame in tab_section:
            fret = frame[i] - 1
            if fret == -1:
                line += '-'
            elif fret < 10:
                line += str(fret)
            else:
                line += chr(ord('A') + fret-10)
        lines.append(line + "\n")
    return lines

weights_path = "saved/c 2025-05-25 14:01:42/0/weights.h5"
predictions = predict_custom_audio("myaudio.wav", weights_path=weights_path)
tab_pred = np.argmax(predictions, axis=-1)

with open("preds.txt", 'w') as f:
    f.writelines(print_tab(tab_pred, 0, len(tab_pred)))
