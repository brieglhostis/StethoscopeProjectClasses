import numpy as np
from scipy import signal
from classes.mfcc import dct as mfcc


class AudioRecording:

    features_names = [
        "Mean",
        "Median",
        "Standard deviation",
        "25th percentile",
        "75th percentile",
        "Mean Absolute Deviation",
        "Inter Quartile Range",
        "Skewness",
        "Kurtosis",
        "Shannon's entropy",
        "Spectral entropy",
        "Maximum frequency",
        "Maximum magnitude",
        "Ratio of signal energy",
        "MFCC 1",
        "MFCC 2",
        "MFCC 3",
        "MFCC 4",
        "MFCC 5",
        "MFCC 6",
        "MFCC 7",
        "MFCC 8",
        "MFCC 9",
        "MFCC 10",
        "MFCC 11",
        "MFCC 12",
        "MFCC 13"
    ]
    F = len(features_names)

    def __init__(self, audio_array, sampling_rate=2000):

        if not isinstance(audio_array, np.ndarray):
            raise TypeError("The recording must be a numpy array")
        if len(audio_array) == 0:
            raise ValueError("Empty recording provided")
        if len(np.shape(audio_array)) != 1:
            raise ValueError("The recording must have exactly one dimension")
        if not (isinstance(sampling_rate, float) or isinstance(sampling_rate, int)):
            raise TypeError("The sampling rate must be a positive float or integer")
        if sampling_rate <= 0:
            raise ValueError("The sampling rate must be a positive float or integer")
        self.raw_signal = audio_array

        self.sampling_rate = sampling_rate
        self.length = len(audio_array)
        self.segments = [audio_array]
        self.one_heart_cycle_segments = [audio_array]
        self.average_heart_cycle = audio_array
        self.is_segmented = False
        self.features = np.zeros((1, self.F))
        self.has_features = False
        self.heart_rate = 60*sampling_rate

    def filter(self, order=6):
        """
        Applies a bandpass filter on the recording, with low cut at 20Hz and high cut at 600Hz.
        :param order: (int) order of the filter
        :return: None
        """

        if not isinstance(order, int):
            raise TypeError("The filter order must be a non negative integer")
        if order < 0:
            raise ValueError("The filter order must be a non negative integer")
        if self.sampling_rate <= 1000:
            raise ValueError("The sampling rate provided is too small to filter the signal")
        low_cut = 20
        high_cut = 600
        sos = signal.butter(order, [low_cut, high_cut], btype='bandpass', fs=self.sampling_rate, output='sos')
        self.raw_signal = signal.sosfilt(sos, self.raw_signal)

    def segment(self, heart_cycles_per_segment=3):
        """
        Applies segmentation to the recording to identify the heart cycles.
        Saves the segments identified as well as the average heart cycle.
        Each segment saved in self.segments include a few heart cycles if possible.
        :param heart_cycles_per_segment: (int) number of heart cycles to include in each segment (if possible)
        :return: None
        """

        if self.is_segmented:
            print("This signal has already been segmented")
            return
        if not isinstance(heart_cycles_per_segment, int):
            raise TypeError("The number of heart cycles per segment must be a positive integer")
        if heart_cycles_per_segment < 1:
            raise ValueError("The number of heart cycles per segment must be a positive integer")
        if heart_cycles_per_segment > self.length:
            raise ValueError("The number of heart cycles per segment must be less than the length of the recording")
        if self.length < self.sampling_rate:
            raise ValueError("The recording provided is shorter than one second with the sampling rate provided and"
                             "thus cannot be segmented properly")

        def segmentation_hilbert_heron(f, length, sampling_rate):
            """
            This segmentation method is an implementation in python of the following article:
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3825056/
            :param f: (float array) sample to be segmented
            :param length: (int) length of the sample
            :param sampling_rate: (float) sampling rate of the sample
            :return: peak_indices (int list), list of the indices corresponding to identified S1 peaks in the sample;
                        heart_rate (float), heart rate (in BPM) deduced from the identified S1 peaks
            """

            def extract_peaks(local_maxima_list, local_minima_list, l):
                """
                This function extracts all peaks from the signal based on its local extrema and defining a peak as a
                succession of a local minimum, a local maximum and a local minimum
                :param local_maxima_list: (int list) list of maxima indices in the signal
                :param local_minima_list: (int list) list of minima indices in the signal
                :param l: (int) length of the signal
                :return: (int tuple array) the list of triplets (previous_minimum, local_maximum, next_minimum)
                for each identified peak
                """

                def find_adjacent_minim(local_maximum_index, local_minima_indices, n):
                    """
                    This function finds the nearest minima to a local maximum
                    :param local_maximum_index: (int) index of a local maximum
                    :param local_minima_indices: (int list) list of minima indices in the signal
                    :param n: (int) length of the signal
                    :return: (int couple) a couple containing the nearest minima
                    """
                    j = 0
                    while j < len(local_minima_indices) and local_minima_indices[j] < local_maximum_index:
                        j += 1
                    if j == 0:
                        return 0, local_minima_indices[j]
                    elif j == len(local_minima_indices):
                        return local_minima_indices[j - 1], n
                    else:
                        return local_minima_indices[j - 1], local_minima_indices[j]

                def find_adjacent_maxim(local_maximum_index, local_maxima_indices, n):
                    """
                    This function finds the nearest maxima to a local maximum
                    :param local_maximum_index: (int) index of a local maximum
                    :param local_maxima_indices: (int list) list of maxima indices in the signal
                    :param n: (int) length of the signal
                    :return: (int couple) a couple containing the nearest maxima
                    """
                    j = 0
                    while j < len(local_maxima_indices) - 1 and local_maxima_indices[j] < local_maximum_index:
                        j += 1
                    if j == 0:
                        return 0, local_maxima_indices[j]
                    elif j == len(local_maxima_indices) - 1:
                        return local_maxima_indices[j - 1], n
                    else:
                        return local_maxima_indices[j - 1], local_maxima_indices[j + 1]

                peak_triplets = []
                for index in range(len(local_maxima)):
                    local_maximum = local_maxima_list[index]
                    previous_minimum, next_minimum = find_adjacent_minim(local_maximum, local_minima_list, l)
                    previous_maxium, next_maximum = find_adjacent_maxim(local_maximum, local_maxima_list, l)

                    if previous_minimum > previous_maxium and next_minimum < next_maximum and previous_minimum != 0 \
                            and next_minimum != length:
                        peak_triplets.append((previous_minimum, local_maximum, next_minimum))

                return peak_triplets

            def apply_heron_triangles_criterion(peak_triplets, envelope):
                """
                This function computes the area of the triangles formed by the peaks using Heron's formula and keeps the
                triangles whose area is higher than the standard deviation of the area list
                :param peak_triplets: (int tuple list) list of the triplets (previous_minimum, local_maximum,
                next_minimum) for each peak
                :param envelope: (float array) hilbert envelope of the signal
                :return: peak_indices_list (int list) list of the indices of the remaining peaks
                """

                p = len(peak_triplets)
                delta_array = np.zeros(p)

                for index in range(p):
                    (previous_minimum, maximum, next_minimum) = peak_triplets[index]

                    a_i = np.sqrt((envelope[maximum] - envelope[previous_minimum]) ** 2 +
                                  (maximum - previous_minimum) ** 2)
                    b_i = np.sqrt((envelope[maximum] - envelope[next_minimum]) ** 2 +
                                  (maximum - next_minimum) ** 2)
                    c_i = np.sqrt((envelope[previous_minimum] - envelope[next_minimum]) ** 2 +
                                  (previous_minimum - next_minimum) ** 2)

                    s_i = (a_i + b_i + c_i) / 2

                    delta_i = np.sqrt(s_i * (s_i - a_i) * (s_i - b_i) * (s_i - c_i))  # Area of the ith triangle
                    delta_array[index] = delta_i

                standard_deviation_delta = np.std(delta_array)

                peak_indices_list = [peak_triplets[x][1] for x in np.where(delta_array > standard_deviation_delta)[0]]

                return peak_indices_list

            def get_s1_peaks(peak_indices_list, envelope):
                """
                Identifies the S1 peaks in a list of peaks. Keeps the peaks that are at least higher than one of their
                neighbor
                :param peak_indices_list: (int list) list of indices of peaks in the signal
                :param envelope: (float array) hilbert envelope of the signal
                :return: prominent_peak_indices (int list), list of the indices of the identified S1 peaks
                """

                prominent_peak_indices = []

                for index in range(1, len(peak_indices_list) - 1):
                    if envelope[peak_indices_list[index]] > envelope[peak_indices_list[index - 1]] or \
                            envelope[peak_indices_list[index]] > envelope[peak_indices_list[index + 1]]:
                        prominent_peak_indices.append(peak_indices_list[index])

                return prominent_peak_indices

            analytical_sample = signal.hilbert(f)
            hilbert_envelope = np.abs(analytical_sample)
            # Apply a 5th order low pass filter of cut frequency 20Hz
            [b, a] = signal.butter(5, 20, 'low', fs=sampling_rate)
            filtered_envelope = signal.filtfilt(b, a, hilbert_envelope)

            local_maxima = signal.argrelextrema(filtered_envelope, np.greater)[0]
            local_minima = signal.argrelextrema(filtered_envelope, np.less)[0]
            peaks = extract_peaks(local_maxima, local_minima, length)
            s1_s2_peak_indices = apply_heron_triangles_criterion(peaks, filtered_envelope)
            s1_peak_indices = get_s1_peaks(s1_s2_peak_indices, filtered_envelope)
            heart_rate_estimation = 60*len(s1_peak_indices)*sampling_rate/length

            return s1_peak_indices, heart_rate_estimation

        def segmentation_autocorrelation(f, length, sampling_rate):
            """
            This segmentation method is uses a modified autocorrelation function to identify S1 peaks
            :param f: (float array) sample to be segmented
            :param length: (int) length of the sample
            :param sampling_rate: (float) sampling rate of the sample
            :return: peak_indices_list (int list), list of the indices corresponding to identified S1 peaks in the
            sample;
                    heart_rate (float), heart rate (in BPM) deduced from the identified S1 peaks
            """

            def auto_correlation(g):
                """
                Computes a modified autocorrelation that is maximum when the heart cycles at compared to themselves
                and not their symmetric.
                :param g: (float Nx1 array) function to autocorrelate
                :return: result from the modified autocorrelation
                """
                n = len(g)
                g = np.concatenate((np.zeros(n), g, np.zeros(n)))
                res = np.zeros(n)
                for j in range(1, n):
                    res[j] = np.dot(g[:-j], g[j:])
                return res

            if length < 4:
                raise ValueError("The recording provided is too short to be segmented")
            cor = auto_correlation(f[:int(length/4)])

            if sampling_rate < 20:
                raise ValueError("The sampling rate provided is too low to segment the recording")
            offset = int(sampling_rate/20)

            length_of_segment = np.argmax(cor[offset:]) + offset
            number_of_peaks = int(length/length_of_segment)  # length_of_segment cannot be 0 after the previous "if"
            peak_indices_list = []
            for index in range(number_of_peaks):
                peak_indices_list.append(
                    index * length_of_segment + np.argmax([f[index*length_of_segment:(index+1)*length_of_segment]]))

            heart_rate_estimation = 60*len(peak_indices)*sampling_rate/length

            return peak_indices_list, heart_rate_estimation

        def segmentation_fixed_distances(f, length, sampling_rate):
            """
            This segmentation method fixes a minimal and maximal time duration between S1 and S2 peaks to identify S1
            peaks
            :param f: (float array) sample to be segmented
            :param length: (int) length of the sample
            :param sampling_rate: (float) sampling rate of the sample
            :return: peak_indices_list (int list), list of the indices corresponding to identified S1 peaks in the
            sample;
                     heart_rate (float), heart rate (in BPM) deduced from the identified S1 peaks
            """
            # Search for S1 peaks that are at least 0.6sec apart (no BPM under 36 to avoid identifying half heart
            # cycles)
            minimum_length_of_heart_cycle = int(sampling_rate * 3 / 5)
            peak_indices_list = [0]
            # At least one peak will be added since self.length > self.sampling rate
            for index in range(minimum_length_of_heart_cycle, length):
                if index - peak_indices_list[-1] > minimum_length_of_heart_cycle:
                    peak_indices_list.append(index)
                else:
                    if f[index] > f[peak_indices_list[-1]]:
                        peak_indices_list[-1] = index

            heart_rate_estimation = 60*len(peak_indices)*sampling_rate/length

            return peak_indices_list, heart_rate_estimation

        peak_indices, heart_rate = segmentation_hilbert_heron(self.raw_signal, self.length, self.sampling_rate)
        if heart_rate < 30 or heart_rate > 220:
            print("hilbert failed", heart_rate)
            peak_indices, heart_rate = segmentation_autocorrelation(self.raw_signal, self.length, self.sampling_rate)
            if heart_rate < 30 or heart_rate > 220:
                print("autocorrelation failed", heart_rate)
                peak_indices, heart_rate = segmentation_fixed_distances(self.raw_signal, self.length,
                                                                        self.sampling_rate)
                if heart_rate < 30 or heart_rate > 220:
                    print("fixed distance segmentation failed", heart_rate)
                    peak_indices, heart_rate = segmentation_hilbert_heron(self.raw_signal, self.length,
                                                                          self.sampling_rate)

        if len(peak_indices) < 3:
            peak_indices = [0, int(self.length/2), self.length]

        self.heart_rate = heart_rate

        average_period = self.length / len(peak_indices)
        offset_before_s1 = int(self.sampling_rate*3/20)
        offset_after_s1 = int(average_period * 0.8)
        offset_after_s1_one_heart_cycle = int(average_period + offset_before_s1)

        self.segments = []

        if peak_indices[0] < offset_before_s1:
            if len(peak_indices) < 1:
                raise ValueError("Only one heart cycle identified in segmentation,"
                                 "the recording provided is either too irregular or too short")
            peak_indices = peak_indices[1:]

        while len(self.segments) == 0 and heart_cycles_per_segment > 0:
            self.one_heart_cycle_segments = []  # recipient of the heart cycles to compute the average heart cycle
            for i in range(len(peak_indices) - heart_cycles_per_segment):
                segment = self.raw_signal[peak_indices[i] - offset_before_s1:
                                          peak_indices[i] + offset_after_s1_one_heart_cycle]
                if len(segment) > 0:
                    self.one_heart_cycle_segments.append(segment)
                if peak_indices[i + heart_cycles_per_segment - 1] + offset_after_s1 < self.length:
                    segment = self.raw_signal[peak_indices[i] - offset_before_s1:
                                              peak_indices[i + heart_cycles_per_segment - 1] + offset_after_s1]
                    if len(segment) > 0:
                        self.segments.append(segment)
            heart_cycles_per_segment -= 1

        if heart_cycles_per_segment == 0 and len(self.segments) == 0:
            raise ValueError("Unexpected error, the segmentation was unsuccessful for an unknown reason")

        if len(self.one_heart_cycle_segments) > 0:
            self.one_heart_cycle_segments = [x[:min([len(one_heart_cycle_segment) for one_heart_cycle_segment in
                                                     self.one_heart_cycle_segments])] for x in
                                             self.one_heart_cycle_segments]
            self.average_heart_cycle = np.mean(self.one_heart_cycle_segments, axis=0)
        else:
            print("Unexpected error, the recording could not be segmented")
            self.one_heart_cycle_segments = np.array([self.raw_signal])
            self.average_heart_cycle = self.raw_signal
        self.is_segmented = True

    def zero_padding(self, heart_cycles_per_segment=3, length_per_heart_cycle=None):
        """
        Formalizes the segments to a fixed length set to (number of cycles per segments)*(sampling rate) by default.
        Cuts segments that are longer than this length and adds zeros to segments that are shorter.
        :param heart_cycles_per_segment: (int) number of heart cycles per segment
        :param length_per_heart_cycle: (int or None) desired length per heart cycle in the segments (if None then the
        length per segment will be set to the default value)
        :return: None
        """

        if length_per_heart_cycle is None:
            length_per_heart_cycle = self.sampling_rate
        else:
            if not isinstance(length_per_heart_cycle, int):
                raise TypeError("The length per heart cycle should be a positive integer")
            if length_per_heart_cycle < 1:
                raise TypeError("The length per heart cycle should be a positive integer")

        def individual_zero_padding(segment, length):
            """
            Applies the zero padding to one segment
            :param segment: (float Lx1 array) segment to be processed
            :param length: (int) length of the processed segment
            :return: (float length x1 array) processed segment
            """
            zero_padded = np.zeros(length)
            for i in range(min(len(segment), length)):
                zero_padded[i] = segment[i]
            return zero_padded

        if not self.is_segmented:
            self.segment(heart_cycles_per_segment=heart_cycles_per_segment)
        self.segments = [individual_zero_padding(segment, int(heart_cycles_per_segment*length_per_heart_cycle)) for
                         segment in self.segments]
        self.one_heart_cycle_segments = [individual_zero_padding(segment, int(length_per_heart_cycle)) for segment in
                                         self.one_heart_cycle_segments]
        self.average_heart_cycle = individual_zero_padding(self.average_heart_cycle, length_per_heart_cycle)

    def extract_features(self, force_segmentation=True):
        """
        Extracts features from the recording, the list of features was decided according to the following article:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6630694/
        :param force_segmentation: (bool) if True then the function will ask to segment the recording before
        extracting the features, else the features will be extracted from the recording in segments (i.e. the
        unprocessed recording)
        :return: features: (float NxF array) array of the features for each segment (N: number of segments, F: number
        of features)
        """

        if not isinstance(force_segmentation, bool):
            raise TypeError("The argument 'force_segmentation' must be a boolean")
        if not isinstance(self.F, int):
            raise TypeError("The number of features should be a positive integer")
        if self.F < 1:
            raise ValueError("The number of features should be a positive integer")

        def individual_extract_features(sample, f=27):
            """
            Extracts the features from one segment
            :param sample: (float Lx1 array) segment to be processed
            :param f: (int) number of features to extract
            :return: features (float Fx1 array) list of the extracted features
            """

            def entropy(elements, normalized=True):
                """
                Computes the entropy of a set of data
                :param elements: (float Nx1 array) list of elements in the set
                :param normalized: (bool) if True then the entropy will be normalized (i.e. divided by log(N)
                where N is the number of elements in the set)
                :return: (float) value of the entropy of the elements
                """
                if not isinstance(normalized, bool):
                    raise TypeError("The argument 'normalized' must be a boolean")
                n = len(elements)
                if n == 0:
                    raise ValueError("Empty set provided to compute the entropy")
                element_range = np.max(elements)-np.min(elements)  # The range of value will be divided into N bands
                distribution = np.zeros(n+1)
                for element in elements:
                    # Identify the band that the element belongs to
                    distribution[int(n*(element-np.min(elements))/element_range)] += 1/n
                distribution = distribution[np.nonzero(distribution)]  # Suppress the empty bands in the distribution
                if normalized:
                    return - np.sum(distribution*np.log(distribution))/np.log(n)
                else:
                    return - np.sum(distribution*np.log(distribution))

            if not isinstance(f, int):
                raise TypeError("The number of features should be a positive integer")
            if f < 1:
                raise ValueError("The number of features should be a positive integer")
            features = np.zeros(f)

            if len(segment) < 2:
                raise ValueError("The segment provided to extract features is too short")

            # Power spectral density of the segment
            frequencies, power_spectral_density = signal.periodogram(sample, self.sampling_rate)

            if len(power_spectral_density) < 1 or np.sum(power_spectral_density) == 0:
                raise ValueError("Unexpected error, invalid power spectral density")

            mean = np.mean(sample)
            median = np.median(sample)
            standard_deviation = np.std(sample)
            if standard_deviation == 0:
                raise ValueError("The standard deviation of the recording over the segment is zero,"
                                 "thus the segment is either empty or constant")
            percentile_25 = np.percentile(sample, 25)
            percentile_75 = np.percentile(sample, 75)
            mad = np.mean(np.abs(sample - mean))  # Mean absolute deviation
            iqr = percentile_75 - percentile_25  # Inter Quartile Range
            skewness = np.mean(np.power((sample - mean) / standard_deviation, 3))  # Measure of the lack of symmetry
            # kurtosis: measure of the sharpness of S1 peaks
            kurtosis = np.mean(np.power((sample - mean) / standard_deviation, 4)) - 3
            try:
                shannon_entropy = entropy(sample)
                spectral_entropy = entropy(power_spectral_density, normalized=True)
            except ValueError:
                raise
            maximum_frequency_index = np.argmax(power_spectral_density)
            maximum_frequency = frequencies[maximum_frequency_index]
            maximum_magnitude = 10*np.log(np.max(power_spectral_density))
            df = 500
            # Ratio of energy between fmax +- df and the whole signal
            ratio_of_signal_energy = np.sum(
                power_spectral_density[max(0, maximum_frequency_index - df):
                                       maximum_frequency_index + df]) / np.sum(power_spectral_density)

            features[:14] = np.array([mean, median, standard_deviation, percentile_25, percentile_75,
                                      mad, iqr, skewness, kurtosis, shannon_entropy, spectral_entropy,
                                      maximum_frequency, maximum_magnitude, ratio_of_signal_energy])

            mel_frequency_coepstrum = mfcc(np.array([power_spectral_density]))[
                0]
            # Computed my the code provided on the following website:
            # http://www.cs.cmu.edu/~dhuggins/Projects/pyphone/sphinx/mfcc.py

            if len(mel_frequency_coepstrum) != 13:
                raise ValueError("Unexpected error, not enough Mel Frequency Coepstrum Coefficients extracted")

            features[14:] = mel_frequency_coepstrum

            return features

        if force_segmentation and not self.is_segmented:
            self.filter()
            self.segment()

        number_of_segments = len(self.segments)
        self.features = np.zeros((number_of_segments, self.F))

        if number_of_segments < 1:
            raise ValueError("Not enough segments, try to set force_segmentation to True")

        for index in range(number_of_segments):
            segment = self.segments[index]
            self.features[index] = individual_extract_features(segment, self.F)

        self.has_features = True

        return self.features

    def get_heart_rate_rhythm(self, heart_cycles_per_segment=3):
        """
        Estimates the heart rate as well as the heart rhythm from the segmented recording.
        :param heart_cycles_per_segment: (int) number of heart cycle per segment to use in the segmentation
        if the recording has not yet been segmented.
        :return: (floats) the estimated heart rate and the rhythm ratio (s1-s2/s1-next_s1).
        """
        if not self.is_segmented:
            self.segment(heart_cycles_per_segment=heart_cycles_per_segment)
        heart_rate = []
        systole = []
        diastole = []
        for segment in self.one_heart_cycle_segments:
            hilbert_envelope = np.abs(signal.hilbert(segment))
            [b, a] = signal.butter(5, 10, 'low', fs=self.sampling_rate)
            filtered_envelope = signal.filtfilt(b, a, hilbert_envelope)
            peaks = signal.argrelextrema(filtered_envelope, np.greater)[0]
            if len(peaks) >= 3:
                max_peaks = peaks[np.argsort(filtered_envelope[peaks])[-3:]]
                s1_peaks = np.sort(max_peaks[-2:]) / self.sampling_rate
                s2_peak = max_peaks[0] / self.sampling_rate
                heart_rate.append(60 / np.abs(s1_peaks[1] - s1_peaks[0]))
                if np.abs(s1_peaks[0] - s2_peak) > np.abs(s1_peaks[1] - s2_peak):
                    systole.append(np.abs(s1_peaks[1] - s2_peak))
                else:
                    systole.append(np.abs(s1_peaks[0] - s2_peak))
                diastole.append(np.abs(s1_peaks[1] - s1_peaks[0]) - systole[-1])
        if not heart_rate:
            return -1, -1
        heart_rate, systole, diastole = np.mean(heart_rate), np.mean(systole), np.mean(diastole)
        if systole > diastole:
            tmp = systole
            systole = diastole
            diastole = tmp
        rhythm = systole/(systole + diastole)
        return heart_rate, rhythm, systole, diastole

    def get_heart_cycle_representation(self, heart_cycles_per_segment=3):
        """
        Returns a couple of functions to plot to represent the recording:
        - average segment: Average of the heart cycles superposed;
        - min segment: Minimum point by point of the heart cycles superposed;
        - max segment: Maximum point by point of the heart cycles superposed;
        - average envelope: Average of the Hilbert envelope of the heart cycles superposed;
        - min envelope: Minimum point by point of the Hilbert envelope of the heart cycles superposed;
        - max envelope: Maximum point by point of the Hilbert envelope of the heart cycles superposed;
        :param heart_cycles_per_segment: (int) number of heart cycle per segment to use in the segmentation
        if the recording has not yet been segmented.
        :return: (average_segment, min_segment, max_segment), (average_envelope, min_envelope, max_envelope):
                    (triplets of float arrays) arrays of the functions defined above
        """

        if not self.is_segmented:
            self.segment(heart_cycles_per_segment=heart_cycles_per_segment)
        max_segment = np.max(self.one_heart_cycle_segments, axis=0)
        min_segment = np.min(self.one_heart_cycle_segments, axis=0)
        average_segment = np.mean(self.one_heart_cycle_segments, axis=0)
        envelopes = [np.abs(signal.hilbert(segment)) for segment in self.one_heart_cycle_segments]
        max_envelope = np.max(envelopes, axis=0)
        min_envelope = np.min(envelopes, axis=0)
        average_envelope = np.mean(envelopes, axis=0)

        return (average_segment, min_segment, max_segment), (average_envelope, min_envelope, max_envelope)
