import streamlit as st
import numpy as np
from scipy.signal import butter, filtfilt, freqz, firwin, lfilter
from scipy.io import wavfile
import io
import matplotlib.pyplot as plt
import librosa
import librosa.display

# --- Fungsi Filter ---

def design_bandstop_filter(order, fs, lowcut, highcut, window_func='hamming'):
    """
    Mendesain filter FIR bandstop berdasarkan parameter yang diberikan.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    coeffs = firwin(order + 1, [low, high], window=window_func, pass_zero='bandpass') # pass_zero='bandpass' for bandstop
    return coeffs

def apply_filter(data, coeffs):
    """
    Menerapkan filter FIR ke data audio.
    """
    filtered_data = lfilter(coeffs, 1.0, data)
    return filtered_data

# --- Streamlit UI ---

st.set_page_config(layout="wide")

st.title("🎧 FIR Bandstop Filter Designer dan Audio Processor")
st.markdown("""
    Desain filter FIR bandstop Anda sendiri, visualisasikan respons frekuensinya, dan terapkan ke sinyal audio.
    Aplikasi ini memungkinkan Anda untuk membuat filter bandstop menggunakan berbagai metode window.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Filter Design Parameters")

    # Filter Type is fixed to bandstop, no selectbox needed
    st.write("Filter Type: **Bandstop**")

    filter_order = st.slider(
        "Filter Order",
        min_value=3,
        max_value=1001,
        value=51,
        step=2,
        key="filter_order_slider"
    )

    sampling_frequency = st.slider(
        "Sampling Frequency (Hz)",
        min_value=8000,
        max_value=96000,
        value=44100,
        step=100,
        key="sampling_freq_slider"
    )

    # Cutoff frequency sliders for bandstop
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        lowcut_freq = st.slider(
            "Lowcut Frequency (Hz)",
            min_value=20,
            max_value=sampling_frequency // 2 - 20,
            value=980,
            key="lowcut_freq_bandstop_slider"
        )
    with col_f2:
        highcut_freq = st.slider(
            "Highcut Frequency (Hz)",
            min_value=lowcut_freq + 10, # Ensure highcut is always greater than lowcut
            max_value=sampling_frequency // 2,
            value=1020,
            key="highcut_freq_bandstop_slider"
        )

    window_function = st.selectbox(
        "Window Function",
        ("hamming", "hann", "blackman", "bartlett", "kaiser", "rectangular"),
        key="window_func_select"
    )

    generate_filter_button = st.button("Generate Filter", key="generate_filter_button")

with col2:
    st.header("1. Filter Design")

    coeffs_placeholder = st.empty()
    freq_resp_placeholder = st.empty()

    if generate_filter_button:
        st.write("Generating filter...")
        try:
            # Design the bandstop filter
            coeffs = design_bandstop_filter(filter_order, sampling_frequency, lowcut_freq, highcut_freq, window_function)

            if coeffs is not None:
                # Display filter coefficients
                coeffs_placeholder.subheader("Filter Coefficients")
                coeffs_data = [{"Index": i, "Value": f"{c:.4f}"} for i, c in enumerate(coeffs)]
                coeffs_placeholder.dataframe(coeffs_data, height=300)
                
                # Download coefficients
                coeffs_csv = io.StringIO()
                np.savetxt(coeffs_csv, coeffs, delimiter=",")
                coeffs_placeholder.download_button(
                    "⬇️ Download Coefficients as CSV",
                    data=coeffs_csv.getvalue(),
                    file_name="filter_coefficients.csv",
                    mime="text/csv"
                )

                # Plot frequency response
                freq_resp_placeholder.subheader("Filter Frequency Response")
                w, h = freqz(coeffs, worN=8000, fs=sampling_frequency)

                fig, axs = plt.subplots(3, 1, figsize=(10, 12))

                # Magnitude Response
                axs[0].plot(w, 20 * np.log10(abs(h)))
                axs[0].set_title('Magnitude Response (dB)')
                axs[0].set_xlabel('Frequency (Hz)')
                axs[0].set_ylabel('Magnitude (dB)')
                axs[0].grid(True)

                # Phase Response
                angles = np.unwrap(np.angle(h))
                axs[1].plot(w, angles)
                axs[1].set_title('Phase Response (rad)')
                axs[1].set_xlabel('Frequency (Hz)')
                axs[1].set_ylabel('Phase (rad)')
                axs[1].grid(True)

                # Group Delay
                group_delay = -np.diff(angles) / np.diff(w)
                axs[2].plot(w[:-1], group_delay)
                axs[2].set_title('Group Delay (samples)')
                axs[2].set_xlabel('Frequency (Hz)')
                axs[2].set_ylabel('Group Delay (samples)')
                axs[2].grid(True)

                plt.tight_layout()
                freq_resp_placeholder.pyplot(fig)
                plt.close(fig) # Close the figure to prevent it from being displayed twice
                
                st.session_state['coeffs'] = coeffs # Store coefficients in session state
                st.session_state['fs_filter'] = sampling_frequency # Store filter fs

        except Exception as e:
            st.error(f"Error generating filter: {e}")

st.markdown("---")

st.header("2. Audio Processing")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"], key="audio_uploader")
st.info("Limit 200MB per file: WAV")

if uploaded_file is not None:
    st.write("Upload an audio file to apply the filter and visualize the results.")
    
    # Check if filter coefficients exist in session state
    if 'coeffs' not in st.session_state or 'fs_filter' not in st.session_state:
        st.warning("Please generate the filter first in 'Filter Design Parameters' section.")
    else:
        fs_audio, audio_data = wavfile.read(uploaded_file)

        # Ensure audio is mono
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        st.write(f"Sample rate: {fs_audio} Hz")

        # Resample audio if sample rate differs from filter design sample rate
        if fs_audio != st.session_state['fs_filter']:
            st.warning(f"Audio sample rate ({fs_audio} Hz) differs from filter design sample rate ({st.session_state['fs_filter']} Hz). Resampling audio...")
            # Using librosa for resampling
            audio_data_resampled = librosa.resample(y=audio_data.astype(float), orig_sr=fs_audio, target_sr=st.session_state['fs_filter'])
            audio_to_process = audio_data_resampled
            fs_process = st.session_state['fs_filter']
        else:
            audio_to_process = audio_data
            fs_process = fs_audio

        st.subheader("Audio Waveforms")
        
        # Plot original audio waveform
        fig_waveform, ax_waveform = plt.subplots(2, 1, figsize=(10, 6))
        ax_waveform[0].plot(np.arange(len(audio_data)) / fs_audio, audio_data)
        ax_waveform[0].set_title('Original Audio Waveform')
        ax_waveform[0].set_xlabel('Time (s)')
        ax_waveform[0].set_ylabel('Amplitude')
        ax_waveform[0].grid(True)

        # Apply filter
        filtered_audio = apply_filter(audio_to_process, st.session_state['coeffs'])

        # Normalization and conversion to int16 for playback/download
        # Ensure filtered_audio is not all zeros or too small to normalize
        if np.max(np.abs(filtered_audio)) > 0:
            filtered_audio_normalized = np.int16(filtered_audio / np.max(np.abs(filtered_audio)) * 32767)
        else:
            filtered_audio_normalized = np.zeros_like(filtered_audio, dtype=np.int16)
            st.warning("Filtered audio has zero amplitude. Check filter parameters.")


        # Plot filtered audio waveform
        ax_waveform[1].plot(np.arange(len(filtered_audio)) / fs_process, filtered_audio)
        ax_waveform[1].set_title('Filtered Audio Waveform')
        ax_waveform[1].set_xlabel('Time (s)')
        ax_waveform[1].set_ylabel('Amplitude')
        ax_waveform[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig_waveform)
        plt.close(fig_waveform)

        st.subheader("Audio Spectrograms")

        # Plot original audio spectrogram
        fig_spec, ax_spec = plt.subplots(2, 1, figsize=(10, 8))
        D_original = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data.astype(float))), ref=np.max)
        librosa.display.specshow(D_original, sr=fs_audio, x_axis='time', y_axis='hz', ax=ax_spec[0])
        ax_spec[0].set_title('Original Audio Spectrogram')
        ax_spec[0].set_xlabel('Time (s)')
        ax_spec[0].set_ylabel('Frequency (Hz)')
        plt.colorbar(ax_spec[0].collections[0], format="%+2.0f dB", ax=ax_spec[0])

        # Plot filtered audio spectrogram
        D_filtered = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio.astype(float))), ref=np.max)
        librosa.display.specshow(D_filtered, sr=fs_process, x_axis='time', y_axis='hz', ax=ax_spec[1])
        ax_spec[1].set_title('Filtered Audio Spectrogram')
        ax_spec[1].set_xlabel('Time (s)')
        ax_spec[1].set_ylabel('Frequency (Hz)')
        plt.colorbar(ax_spec[1].collections[0], format="%+2.0f dB", ax=ax_spec[1])

        plt.tight_layout()
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        st.subheader("Audio Playback")
        col_original_audio, col_filtered_audio = st.columns(2)
        with col_original_audio:
            st.write("Original Audio")
            st.audio(uploaded_file, format='audio/wav')
        with col_filtered_audio:
            st.write("Filtered Audio")
            # Save filtered audio to buffer
            buffer_filtered = io.BytesIO()
            wavfile.write(buffer_filtered, fs_process, filtered_audio_normalized)
            st.audio(buffer_filtered, format='audio/wav')
            st.download_button(
                "⬇️ Download Filtered Audio",
                data=buffer_filtered.getvalue(),
                file_name="filtered_audio.wav",
                mime="audio/wav",
                key="download_filtered_audio_button"
            )

st.markdown("---")
st.markdown("### About FIR Filters")
st.markdown("FIR Filter Designer and Audio Processor")
st.markdown("Created with Streamlit, SciPy, Matplotlib, and Librosa.")
