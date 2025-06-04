import librosa
import numpy as np
import soundfile as sf
import pyworld as pw  # Alternative to crepe for F0 extraction

print("=" * 50)
print("WaveTopo Acoustic Topology Optimizer v1.0 (Compatible Version)")
print("=" * 50)

def enhance_waveform(y, f0, confidence, sr):
    """
    Core algorithm for acoustic topology optimization
    """
    # Compute STFT
    stft = librosa.stft(y)
    mag = np.abs(stft)
    
    # Calculate valid fundamental frequency (remove low confidence parts)
    valid_mask = confidence > 0.5
    mean_f0 = np.mean(f0[valid_mask]) if np.any(valid_mask) else 200
    
    # Topology optimization formula (simulates vocal adaptation)
    formant_shift = 1.0 + (mean_f0 - 180) / 800
    
    # Create spectral enhancer
    freq_bins = librosa.fft_frequencies(sr=sr)
    enhancer = 1 + 0.7 * np.exp(-freq_bins/(formant_shift * 4000))
    
    # Apply filter
    mag_enhanced = mag * enhancer[:, np.newaxis]
    
    # Reconstruct waveform
    stft_enhanced = mag_enhanced * np.exp(1j * np.angle(stft))
    return librosa.istft(stft_enhanced)

# ===== Main Program =====
try:
    # 1. Load audio
    y, sr = librosa.load('test.wav', sr=44100, mono=True)
    print(f"Audio loaded successfully | Duration: {len(y)/sr:.2f} seconds")
    
    # 2. Fundamental frequency detection (using pyworld instead of crepe)
    # Extract fundamental frequency
    f0, t = pw.dio(y.astype(np.float64), sr, frame_period=10)
    # Refine F0
    f0 = pw.stonemask(y.astype(np.float64), f0, t, sr)
    confidence = np.ones_like(f0)  # Use all-ones array as confidence
    
    # Ensure length matches through interpolation
    expected_frames = len(y) // 512  # Number of STFT frames
    if len(f0) < expected_frames:
        f0 = np.interp(
            np.linspace(0, 1, expected_frames),
            np.linspace(0, 1, len(f0)),
            f0
        )
    
    print(f"Fundamental frequency analysis complete | Average F0: {np.mean(f0):.1f} Hz")
    
    # 3. Apply topology optimization
    enhanced_audio = enhance_waveform(y, f0, confidence, sr)
    print("Acoustic topology optimization complete!")
    
    # 4. Save result
    sf.write('test_enhanced.wav', enhanced_audio, sr)
    print("Enhanced audio saved: test_enhanced.wav")
    print("Open this file in UTAU and test with pitch shifting")
    
    # 5. Automatically open results folder
    import os
    os.startfile(os.getcwd())  # Open folder automatically
    
except Exception as e:
    print("Error occurred:", str(e))
    import traceback
    traceback.print_exc()
    print("Please ensure:")
    print("1) test.wav file exists")
    print("2) Internet connection is available")
    input("Press Enter to exit...")