# Music AI

Automatic Music Transcription (AMT) has ad-
vanced significantly for the piano, but transcrip-
tion for the guitar remains limited due to several
key challenges. Existing systems fail to detect
and annotate expressive techniques (e.g., slides,
bends, percussive hits) and map notes to the in-
correct string and fret combination in the gen-
erated tablature. Furthermore, prior models are
typically trained on professionally recorded, iso-
lated datasets, limiting their generalizability to
varied acoustic environments with background
noise such as home recordings made on standard
smartphones. To overcome these limitations, we
propose TART, a four-stage end-to-end pipeline
that produces detailed guitar tablature directly
from guitar audio. Our system consists of (1) a
CRNN-based audio-to-MIDI transcription model;
(2) a CNN-BiLSTM for expressive technique clas-
sification; (3) a Transformer-based string and fret
assignment model; and (4) an automated tablature
generator, all consolidated into a pipeline that can
output tablature from a given audio sample. To
the best of our knowledge, this framework is the
first to generate detailed tablature sheet music
with accurate fingerings and expressive technique
labels from guitar audio.
