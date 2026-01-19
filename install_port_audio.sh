# 1. Install PortAudio using Homebrew
brew install portaudio

# 2. Tell the compiler where the PortAudio files are located
export CFLAGS="-I$(brew --prefix portaudio)/include"
export LDFLAGS="-L$(brew --prefix portaudio)/lib"

# 3. Install PyAudio
pip install pyaudio
