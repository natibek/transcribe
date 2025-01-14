from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor, CNNChordFeatureProcessor, CRFChordRecognitionProcessor
from madmom.features.notes import NotePeakPickingProcessor, RNNPianoNoteProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.processors import SequentialProcessor
import yt_dlp
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_audio_to_tempfile(youtube_url):
    #with tempfile.TemporaryDirectory(delete=False) as temp_dir:
    # temp_dir = tempfile.mkdtemp()
    # temp_file_name = os.path.join(temp_dir, "audio")

    # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    # temp_file_name = temp_file.name
    # temp_file.close()  
    temp_file_name = "testing" 
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': temp_file_name,
        'quiet': True,
    }
    print(temp_file_name) 
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return temp_file_name

CHORDREC = SequentialProcessor([DeepChromaProcessor(), DeepChromaChordRecognitionProcessor()])

note_processor= SequentialProcessor([RNNPianoNoteProcessor(), NotePeakPickingProcessor(fps=100, pitch_offset=21)])
NOTEREC = lambda audio: [(note[0], midi_to_note(note[-1])) for note in note_processor(audio)]
key_processor = CNNKeyRecognitionProcessor()
KEYREC = lambda audio: key_prediction_to_label(key_processor(audio))

def detect_chords_deep(audio_file):
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chordrec = SequentialProcessor([dcp, decode])
    chords = chordrec(audio_file)
    for start, end, chord in chords:
        print(f"{start:.2f} - {end:.2f} => {chord}")
    return chords

def detect_chords_crf(audio_file):
    featproc = CNNChordFeatureProcessor()
    decode = CRFChordRecognitionProcessor()
    chordrec = SequentialProcessor([featproc, decode])
    chords = chordrec(audio_file)
    for start, end, chord in chords:
        print(f"{start:.2f} - {end:.2f} => {chord}")
    # for timestamp, chord in chords:
    #     print(f"Timestamp: {timestamp:.2f}sec, Chord: {chord}")
    return chords

def detect_notes(audio_file):
    rpnp = RNNPianoNoteProcessor()
    decode = NotePeakPickingProcessor(fps=100, pitch_offset=21)
    noterec = SequentialProcessor([rpnp, decode])
    notes = noterec(audio_file)
    for timestamp, note in notes:
        print(f"{timestamp:.2f} => {note} {midi_to_note(note)}")
    return notes

def detect_tempo(audio_file):
    pass
    # MAYBE

def midi_to_note(midi_number):
    midi_number = int(midi_number)
    if 21 > midi_number or midi_number > 127:
        raise ValueError(f"Invalid midi number `{midi_number}`. Expected value between 21 and 127.")
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note_name = notes[midi_number % 12]
    return f"{note_name}{octave}"

def detect_key(audio_file):
    ckrp = CNNKeyRecognitionProcessor()
    predictions = ckrp(audio_file)
    key = key_prediction_to_label(predictions)
    print(f"{key=}")
    return key


def transcribe(audio_file):
    NUM_THREADS = 3
    detectors = [KEYREC, NOTEREC, CHORDREC]
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(detector, audio_file) for detector in detectors]
        results = []

        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(f"Error: {e}")


    return results

    

if __name__ == "__main__":
    # youtube_url = "https://www.youtube.com/watch?v=FE_KnGLsX5c&ab_channel=AndricAlejandroVargasAlarcon"
    # temp_audio_path = download_audio_to_tempfile(youtube_url)
    # print(f"Audio downloaded to temporary file: {temp_audio_path}")

    audio_file = "test.mp3" # temp_audio_path # "audio_files/dont_let_me_down.mp3"

    # print("Detect key")
    # key = detect_key(audio_file)

    # print("DeepChromaProcessor")
    # chords = detect_chords_deep(audio_file)
    # # print("CNNChordFeatureProcessor")
    # # chords2 = detect_chords_crf(audio_file)

    # print("Detect Notes")
    # notes = detect_notes(audio_file)

    key, notes, chords = transcribe(audio_file)

    print("key", key_prediction_to_label(key))
    print("chords", chords)
    print("notes", notes)

    exit()
    


