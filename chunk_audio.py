#!/usr/bin/env python3
import os
import argparse
import numpy as np
from scipy.io import wavfile

def get_energy_profile(data, frame_length, hop_length):
    """Computes the Root Mean Square (RMS) energy over sliding windows."""
    # Convert stereo to mono by averaging channels if necessary
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Prevent integer overflow during squaring by converting to float64
    data_float = data.astype(np.float64)

    num_frames = (len(data_float) - frame_length) // hop_length + 1
    energies = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        window = data_float[start:end]
        energies[i] = np.sqrt(np.mean(window**2) + 1e-10)

    return energies

def find_split_indices(energies, num_splits, hop_length, max_samples):
    """Finds the optimal threshold to get exactly N chunks using binary search."""
    min_energy, max_energy = np.min(energies), np.max(energies)
    low, high = min_energy, max_energy

    best_cuts = []

    # Binary search over thresholds to find exactly (num_splits - 1) silent cuts
    for _ in range(60):
        mid_threshold = (low + high) / 2
        is_silent = energies < mid_threshold

        # Track where audio transitions from non-silent to silent
        cuts = []
        in_silence = False

        for idx, silent in enumerate(is_silent):
            if silent and not in_silence:
                cuts.append(idx * hop_length)
                in_silence = True
            elif not silent:
                in_silence = False

        # We need exactly N-1 cut points to make N splits
        if len(cuts) == num_splits - 1:
            best_cuts = cuts
            break
        elif len(cuts) >= num_splits:
            # Too many cuts found -> reduce threshold to look for quieter parts
            high = mid_threshold
        else:
            # Too few cuts found -> increase threshold to accept louder background noise
            low = mid_threshold

    # Fallback if binary search failed to hit the exact target count
    if len(best_cuts) != num_splits - 1:
        print("⚠️ Exact silent boundaries not found. Defaulting to uniform time cuts.")
        step = max_samples // num_splits
        best_cuts = [step * i for i in range(1, num_splits)]

    # Append start and end markers to finalize segment boundaries
    return [0] + best_cuts + [max_samples]

def main():
    parser = argparse.ArgumentParser(
        description="Lightweight CLI tool to split a long WAV file into exactly N parts on silences."
    )
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to input WAV file")
    parser.add_argument("-n", "--splits", required=True, type=int, help="Target number of output splits")
    parser.add_argument("-o", "--output-dir", type=str, default="splits", help="Directory to save generated chunks")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"⚡ Loading {args.input}...")
    sr, data = wavfile.read(args.input)

    # Configure frames (20ms frames, 10ms hops)
    frame_length = int(sr * 0.02)
    hop_length = int(sr * 0.01)

    print(f"🔍 Analyzing energy to locate {args.splits} optimal silent boundaries...")
    energies = get_energy_profile(data, frame_length, hop_length)
    boundaries = find_split_indices(energies, args.splits, hop_length, len(data))

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    print(f"💾 Slicing and exporting chunks...")
    for i in range(args.splits):
        start_sample = boundaries[i]
        end_sample = boundaries[i+1]

        chunk = data[start_sample:end_sample]
        output_filename = f"{base_name}_part_{i+1:03d}.wav"
        output_path = os.path.join(args.output_dir, output_filename)

        wavfile.write(output_path, sr, chunk)

        start_sec = start_sample / sr
        end_sec = end_sample / sr
        print(f" Saved: {output_path} [{start_sec:.2f}s -> {end_sec:.2f}s]")

    print("✅ Complete!")

if __name__ == "__main__":
    main()
