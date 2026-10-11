
import math
import platform
import queue
import sys

import numpy as np
import pygame
from proctap import ProcessAudioCapture



# Windows 11 application-specific audio capture only.
# No microphone or sounddevice input is used.

SAMPLE_RATE = 48000
CHANNELS = 2

CHUNK = 512
ANALYSIS_SIZE = 10240

BARS = 64
WIDTH = 800
HEIGHT = 600

BIN_FLOOR = 0.1
MIN_NORM = 10.0
FLOOR_OFFSET = 40
GAIN = 2.0
SMOOTHING = 0.80

AUDIO_QUEUE_SIZE = 64


def log_bins(spectrum, bars, sample_rate, chunk_size, bin_floor):
    freqs = np.fft.rfftfreq(
        chunk_size, 1.0 / sample_rate
    )

    min_freq = 40
    max_freq = min(10000, freqs[-1])

    edges = np.logspace(
        math.log10(min_freq),
        math.log10(max_freq),
        bars + 1,
    )

    values = []

    for i in range(bars):
        indices = np.where(
            (freqs >= edges[i]) &
            (freqs < edges[i + 1])
        )[0]

        if len(indices):
            values.append(
                max(float(np.mean(spectrum[indices])), bin_floor)
            )
        else:
            values.append(bin_floor)

    return values


def draw_visualizer(
    screen,
    window_size,
    heights,
    bars,
    floor_offset,
):
    width, height = window_size
    bar_width = max(1, width // bars)
    floor = height - floor_offset

    screen.fill((0, 0, 0))

    pygame.draw.line(
        screen,
        (200, 200, 200),
        (0, floor),
        (width, floor),
        2,
    )

    for index, bar_height in enumerate(heights):
        bar_height = min(max(0, bar_height), floor)
        x = index * bar_width
        y = max(0, floor - bar_height)

        pygame.draw.rect(
            screen,
            (0, 255, 0),
            (
                x,
                y,
                max(1, bar_width - 2),
                bar_height,
            ),
        )

    pygame.display.flip()


def main():
    if platform.system() != "Windows":
        raise RuntimeError(
            "This version is intended for Windows 11."
        )

    print("Application Audio Visualizer")
    print("-----------------------------")
    print("Captures audio from one application.")
    print("Microphone input is not used.")
    print()
    print("Find the application's PID in Task Manager:")
    print("Task Manager -> Details -> PID")
    print()

    try:
        target_pid = int(
            input("Enter target application PID: ").strip()
        )
    except ValueError:
        raise SystemExit("Invalid PID. Please enter a number.")

    if target_pid <= 0:
        raise SystemExit("PID must be a positive number.")

    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

    def on_audio(pcm, frames):
        # ProcTap supplies captured PCM as float32 bytes.
        # This callback does not access a microphone.
        try:
            samples = np.frombuffer(
                pcm, dtype=np.float32
            ).copy()

            usable = (samples.size // CHANNELS) * CHANNELS
            if usable == 0:
                return

            samples = samples[:usable].reshape(-1, CHANNELS)

            try:
                audio_queue.put_nowait(samples)
            except queue.Full:
                # Discard the oldest block to keep latency low.
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    audio_queue.put_nowait(samples)
                except queue.Full:
                    pass

        except (ValueError, TypeError):
            # Ignore malformed or incomplete audio blocks.
            pass

    pygame.init()
    pygame.display.set_caption(
        "Arkam's Audio Visualizer - App Audio Only"
    )

    window_size = [WIDTH, HEIGHT]
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT), pygame.RESIZABLE
    )

    buffer = np.zeros(ANALYSIS_SIZE, dtype=np.float32)
    previous_heights = [0] * BARS

    pending = np.empty((0, CHANNELS), dtype=np.float32)

    clock = pygame.time.Clock()
    running = True
    tap = None

    try:
        
         # The target PID is the only audio capture source.
        tap = ProcessAudioCapture(
             pid=target_pid,
             on_data=on_audio,)

        tap.start()

        print(f"Capturing application PID {target_pid}.")
        print("Close the visualizer window to stop.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    window_size[0] = max(200, event.w)
                    window_size[1] = max(150, event.h)

                    screen = pygame.display.set_mode(
                        tuple(window_size),
                        pygame.RESIZABLE,
                    )

            # Transfer received audio blocks into our analysis buffer.
            try:
                while True:
                    block = audio_queue.get_nowait()
                    pending = np.concatenate(
                        (pending, block), axis=0
                    )
            except queue.Empty:
                pass

            # Process fixed-size blocks on the Pygame thread.
            while len(pending) >= CHUNK:
                block = pending[:CHUNK]
                pending = pending[CHUNK:]

                mono = np.mean(block, axis=1)

                buffer[:-CHUNK] = buffer[CHUNK:]
                buffer[-CHUNK:] = mono

                # Window and FFT over the rolling audio buffer.
                window = np.hanning(ANALYSIS_SIZE)
                windowed = buffer * window

                spectrum = (
                    np.abs(np.fft.rfft(windowed)) * GAIN
                )
                spectrum = np.power(spectrum, 0.7)

                values = log_bins(
                    spectrum,
                    BARS,
                    SAMPLE_RATE,
                    ANALYSIS_SIZE,
                    BIN_FLOOR,
                )

                max_value = max(max(values), MIN_NORM)

                heights = [
                    max(
                        0,
                        int(
                            value / max_value
                            * (window_size[1] - FLOOR_OFFSET)
                        ),
                    )
                    for value in values
                ]

                heights = [
                    int(
                        SMOOTHING * old
                        + (1.0 - SMOOTHING) * new
                    )
                    for old, new in zip(
                        previous_heights, heights
                    )
                ]

                previous_heights[:] = heights

                draw_visualizer(
                    screen,
                    window_size,
                    heights,
                    BARS,
                    FLOOR_OFFSET,
                )

            clock.tick(120)

    except KeyboardInterrupt:
        pass

    finally:
        if tap is not None:
            try:
                tap.close()
            except Exception as exc:
                print(f"Audio capture cleanup warning: {exc}")

        pygame.quit()


if __name__ == "__main__":
    main()
