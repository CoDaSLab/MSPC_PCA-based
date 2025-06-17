from obspy import Trace, Stream
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt

def find_window_index(time, start_time, window_duration, overlap=0):
    """
    Encuentra el índice de la ventana que contiene un tiempo específico.
    
    :param time: Tiempo a buscar (UTCDateTime)
    :param start_time: Tiempo inicial de la serie (UTCDateTime)
    :param window_duration: Duración de cada ventana en segundos
    :param overlap: Fracción de solapamiento entre ventanas
    :return: Lista de índices de ventanas que contienen el tiempo
    """
    if isinstance(time, str):
        time = UTCDateTime(time)
    
    elapsed_time = time - start_time
    
    step = window_duration * (1 - overlap)
    
    if overlap > 0:
        potential_windows = []
    
    window_index = int(elapsed_time / step)
    window_start = start_time + (window_index * step)
    window_end = window_start + window_duration
    
    if window_start <= time <= window_end:
        if overlap > 0:
            potential_windows.append(window_index)
        else:
            return window_index
    
    # Debido al solapamiento, también podría estar en la ventana anterior
    if window_index > 0:
        prev_window_start = start_time + ((window_index-1) * step)
        prev_window_end = prev_window_start + window_duration
        if prev_window_start <= time <= prev_window_end:
            potential_windows.append(window_index-1)
    
    return sorted(potential_windows)


def gaps_indices(start_time, end_time, series_start_time, window_duration, overlap=0):
    """
    Encuentra todas las ventanas que están completamente o parcialmente dentro de un gap.

    :param start_time: Tiempo de inicio del gap (UTCDateTime).
    :param end_time: Tiempo de fin del gap (UTCDateTime).
    :param series_start_time: Tiempo inicial de la serie (UTCDateTime).
    :param window_duration: Duración de cada ventana en segundos.
    :param overlap: Fracción de solapamiento entre ventanas.
    :return: Lista de índices de ventanas que abarcan el gap.
    """
    if isinstance(start_time, str):
        start_time = UTCDateTime(start_time)
    if isinstance(end_time, str):
        end_time = UTCDateTime(end_time)
    if isinstance(series_start_time, str):
        series_start_time = UTCDateTime(series_start_time)

    step = window_duration * (1 - overlap)

    first_window_index = int((start_time - series_start_time) / step)
    last_window_index = int((end_time - series_start_time) / step)

    return list(range(first_window_index, last_window_index + 1))

def calculate_gap_windows(streams, start, end, window_duration, overlap=0):
    """
    Calcula los índices de las ventanas afectadas por gaps en una lista de streams.

    :param streams: Lista de objetos Stream de ObsPy.
    :param start: Fecha de inicio de la serie (UTCDateTime o str).
    :param end: Fecha de fin de la serie (UTCDateTime o str).
    :param window_duration: Duración de cada ventana en segundos.
    :param overlap: Fracción de solapamiento entre ventanas (por defecto 0).
    :return: Lista de índices de ventanas afectadas por gaps.
    """
    if isinstance(start, str):
        start = UTCDateTime(start)
    if isinstance(end, str):
        end = UTCDateTime(end)

    # Obtener los tiempos de inicio y fin de los gaps de todos los streams
    start_end_gaps = []
    for stream in streams:
        gaps = stream.get_gaps()
        start_end_gaps.extend([(gap[4], gap[5]) for gap in gaps])

    start_end_gaps = [(UTCDateTime(s), UTCDateTime(e)) for s, e in start_end_gaps]

    # Calcular a qué ventanas pertenecen los gaps
    gap_window_indices = []
    for s, e in start_end_gaps:
        windows = gaps_indices(s, e, series_start_time=start, window_duration=window_duration, overlap=overlap)
        gap_window_indices.extend(windows)

    # Eliminar duplicados y ordenar los índices
    gap_window_indices = sorted(set(gap_window_indices))

    return gap_window_indices

def get_window_time_range(index, start_time, window_duration, overlap=0):
    """
    Obtiene el rango de tiempo de una ventana dado su índice.

    :param index: Índice de la ventana
    :param start_time: Tiempo inicial de la serie (UTCDateTime)
    :param window_duration: Duración de cada ventana en segundos
    :param overlap: Fracción de solapamiento entre ventanas
    :return: Tupla con el tiempo de inicio y fin de la ventana (UTCDateTime, UTCDateTime)
    """
    step = window_duration * (1 - overlap)
    window_start = start_time + (index * step)
    window_end = window_start + window_duration
    return window_start, window_end


def windowing(tr, seconds, overlap=0):
    """
    Divide una traza en ventanas con solapamiento.
    
    :param tr: Traza de ObsPy
    :param seconds: Duración de cada ventana en segundos
    :param overlap: Fracción de solapamiento entre ventanas (0 a 1)
    :return: Stream con las ventanas
    """
    fs = tr.stats.sampling_rate 
    win_len = seconds
    npts_per_window = int(fs * win_len)
    npts_total = tr.stats.npts
    
    step = int(npts_per_window * (1 - overlap))
    
    windows = []
    
    for i in range(0, npts_total - npts_per_window + 1, step):
        win_data = tr.data[i : i + npts_per_window]

        # Creamos una nueva traza con los datos de la ventana
        new_stats = tr.stats.copy()
        new_stats.starttime = tr.stats.starttime + (i / fs)
        new_stats.npts = npts_per_window
        
        win_trace = Trace(data=win_data, header=new_stats)
        windows.append(win_trace)
    
    windows_stream = Stream(windows)
    print(f"Creadas {len(windows)} ventanas con {overlap*100}% de solapamiento")
    
    return windows_stream

def stream_to_fft(st, n_bands=1800):
    """
    Versión vectorizada mejorada: aplica FFT a cada traza de un Stream y promedia en bandas.

    :param st: ObsPy Stream con ventanas (todas de igual longitud)
    :param n_bands: Número de bandas de frecuencia
    :return: matriz (n_ventanas, n_bandas)
    """
    fs = st[0].stats.sampling_rate
    npts = st[0].stats.npts

    data_matrix = np.stack([tr.data.astype(np.float64) for tr in st])
    
    # Demean
    data_matrix -= np.mean(data_matrix, axis=1, keepdims=True)
    
    # FFT
    fft_vals = np.fft.rfft(data_matrix, axis=1) / npts
    mag = np.abs(fft_vals)
    
    # Frecuencias y bandas
    freqs = np.fft.rfftfreq(npts, d=1/fs)
    max_freq = fs / 2
    band_edges = np.linspace(0, max_freq, n_bands + 1)
    
    # Crear máscaras de banda
    band_masks = [(freqs >= band_edges[i]) & (freqs < band_edges[i + 1]) 
                 for i in range(n_bands)]
    band_masks = np.array(band_masks)
    
    band_sums = band_masks.sum(axis=1)
    #band_sums[band_sums == 0] = 1  # Evitar divisiones por cero
    result = np.dot(mag, band_masks.T) / band_sums
    
    return result


def plot_fft(fft_matrix, channel, fs=100, log_scale=True):
    """
    Visualiza la matriz FFT como un espectrograma de amplitudes.
    
    :param fft_matrix: Matriz de amplitudes FFT
    :param fs: Frecuencia de muestreo
    :param n_bands: Número de bandas de frecuencia
    :param log_scale: Si True, muestra en escala logarítmica (dB)
    """
    plt.figure(figsize=(12, 6))
    
    if log_scale:
        # Convertir a decibelios
        data_to_plot = np.log10(fft_matrix + 1e-12) * 20 
        amplitude_label = 'Amplitud (dB)'
    else:
        data_to_plot = fft_matrix
        amplitude_label = 'Amplitud (counts)'
    
    plt.imshow(
        data_to_plot.T,
        aspect='auto',
        origin='lower',
        extent=[0, fft_matrix.shape[0], 0, fs/2],
        cmap='viridis'
    )
    plt.colorbar(label=amplitude_label)
    plt.xlabel('Ventana temporal')
    plt.ylabel('Frecuencia (Hz)')
    plt.title(f'Espectrograma {channel}')
    plt.tight_layout()
    plt.show()