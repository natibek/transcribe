from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor, CNNChordFeatureProcessor, CRFChordRecognitionProcessor
from madmom.processors import SequentialProcessor

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
    pass


if __name__ == "__main__":
    audio_file = "dont_let_me_down.mp3"
    print("DeepChromaProcessor")
    chords = detect_chords_deep(audio_file)
    print("CNNChordFeatureProcessor")
    # chords2 = detect_chords_crf(audio_file)


