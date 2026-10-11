
"""
Audio-Reactive Particle Visualizer
----------------------------------
Platform: Windows 11
Libraries: pygame, numpy, proc-tap

Features:
- Captures audio from a selected application PID.
- Does not use the microphone.
- Fullscreen particle animation with a green theme.
- Bass-reactive rings and particle movement.
- Audio diagnostics every five seconds.

Controls:
    F11 = Toggle fullscreen/windowed
    H   = Hide/show information overlay
    R   = Reset particles
    ESC = Quit

IMPORTANT:
The audio decoder below assumes float32 stereo at 48 kHz.
If the diagnostics show zeros during playback, verify the
actual audio format returned by your installed proc-tap.
"""

import math
import platform
import queue
import threading
import time

import numpy as np
import pygame
from proctap import ProcessAudioCapture


# ============================================================
# 1. CONFIGURATION
# ============================================================

SAMPLE_RATE = 48000       # Expected audio sample rate
CHANNELS = 2              # Expected stereo: left and right
AUDIO_QUEUE_SIZE = 64     # Maximum queued audio blocks
LOG_INTERVAL = 5.0        # Seconds between diagnostic messages
PARTICLE_COUNT = 400      # Number of particles on screen
FPS = 60                  # Target frames per second

ANALYSIS_SIZE = 4096      # Audio samples used for FFT analysis
BASS_MIN_HZ = 40          # Lower bass frequency boundary
BASS_MAX_HZ = 250         # Upper bass frequency boundary

# RGB colours: each channel ranges from 0 to 255.
BLACK = (0, 0, 0)
GREEN = (0, 255, 75)
DIM_GREEN = (0, 100, 35)
TEXT_GREEN = (170, 190, 170)


# ============================================================
# 2. SMALL HELPER FUNCTIONS
# ============================================================

def clamp(value, low, high):
    """Keep a number inside a specified range."""
    return max(low, min(high, value))


def new_particle(rng, width, height):
    """
    Create one particle with random movement properties.

    angle  = direction around the centre, in radians
    radius = distance from the centre
    speed  = outward movement speed
    size   = particle radius in pixels
    phase  = offset used for subtle animated movement
    """
    return {
        "angle": float(rng.uniform(0, math.tau)),
        "radius": float(
            rng.uniform(20, min(width, height) * 0.45)
        ),
        "speed": float(rng.uniform(15, 100)),
        "size": int(rng.choice([1, 1, 1, 2, 2, 3])),
        "phase": float(rng.uniform(0, math.tau)),
    }


# ============================================================
# 3. MAIN PROGRAM
# ============================================================

def main():

    # This version uses Windows-specific process audio capture.
    if platform.system() != "Windows":
        raise RuntimeError(
            "This visualizer is intended for Windows 11."
        )

    print("=" * 55)
    print("AUDIO-REACTIVE PARTICLE VISUALIZER")
    print("=" * 55)
    print("Audio source: selected application PID")
    print("Microphone: not used")
    print()
    print("Find the target application's PID:")
    print("Task Manager -> Details -> PID")
    print()

    # Ask which application's audio should be captured.
    try:
        target_pid = int(
            input("Enter target application PID: ").strip()
        )
    except ValueError:
        raise SystemExit("Please enter a valid numeric PID.")

    if target_pid <= 0:
        raise SystemExit("PID must be a positive number.")

    # --------------------------------------------------------
    # Shared audio data
    # --------------------------------------------------------

    # The callback thread places audio blocks into this queue.
    # The Pygame thread consumes them.
    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

    # Store the latest callback information for diagnostic logs.
    latest_info = {"data": None}

    # Protect shared diagnostic data from simultaneous access.
    info_lock = threading.Lock()

    # --------------------------------------------------------
    # 4. AUDIO CALLBACK
    # --------------------------------------------------------

    def on_audio(pcm, frames):
        """
        Called by proc-tap when an audio block arrives.

        pcm   = raw audio bytes
        frames = frame count reported by proc-tap

        This decoder assumes float32 stereo audio. Confirm
        the package's actual format if samples appear to be zero.
        """

        try:
            byte_count = len(pcm)

            # Decode the bytes as 32-bit floating-point values.
            samples = np.frombuffer(
                pcm, dtype=np.float32
            ).copy()

            # Keep only complete stereo frames.
            usable = (
                samples.size // CHANNELS
            ) * CHANNELS

            if usable == 0:
                return

            # Convert [L, R, L, R, ...] into rows of [L, R].
            samples = samples[:usable].reshape(-1, CHANNELS)

            # Peak indicates the largest absolute sample.
            peak = float(np.max(np.abs(samples)))

            # RMS estimates the overall signal level.
            rms = float(np.sqrt(np.mean(samples ** 2)))

            # Record diagnostics for the main thread.
            info = {
                "time": time.time(),
                "callback_frames": frames,
                "bytes": byte_count,
                "float_values": int(samples.size),
                "decoded_frames": int(len(samples)),
                "peak": peak,
                "rms": rms,
                "first_samples": (
                    samples.flatten()[:10].tolist()
                ),
            }

            with info_lock:
                latest_info["data"] = info

            # Add the latest audio block without blocking capture.
            try:
                audio_queue.put_nowait(samples)

            except queue.Full:
                # Drop an old block to reduce audio-analysis delay.
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass

                # Attempt to enqueue the newest block.
                try:
                    audio_queue.put_nowait(samples)
                except queue.Full:
                    pass

        except (ValueError, TypeError, BufferError) as exc:
            print(f"[AUDIO] Callback decode error: {exc}")

    # ========================================================
    # 5. INITIALISE PYGAME
    # ========================================================

    pygame.init()
    pygame.display.set_caption(
        "Audio-Reactive Particles - Application Audio Only"
    )

    # Open the visualizer in fullscreen mode.
    display_info = pygame.display.Info()

    screen = pygame.display.set_mode(
        (display_info.current_w, display_info.current_h),
        pygame.FULLSCREEN,
    )

    width, height = screen.get_size()

    # The centre is the origin for the particle animation.
    cx = width / 2
    cy = height / 2

    clock = pygame.time.Clock()

    # Fonts used by the optional information overlay.
    font = pygame.font.Font(None, 27)
    small_font = pygame.font.Font(None, 20)

    # Use a random-number generator for particle properties.
    rng = np.random.default_rng()

    # Create the initial particle field.
    particles = [
        new_particle(rng, width, height)
        for _ in range(PARTICLE_COUNT)
    ]

    # ========================================================
    # 6. AUDIO ANALYSIS STATE
    # ========================================================

    # Rolling buffer containing the most recent mono samples.
    audio_buffer = np.zeros(
        ANALYSIS_SIZE, dtype=np.float32
    )

    # Blocks may not arrive in multiples of 512 samples.
    # Keep incomplete blocks here until enough samples arrive.
    pending = np.empty(
        (0, CHANNELS), dtype=np.float32
    )

    # Raw and smoothed bass measurements.
    bass_level = 0.0
    smoothed_bass = 0.0

    # Runtime state.
    fullscreen = True
    show_hud = True
    running = True
    tap = None

    last_log_time = time.monotonic()

    # ========================================================
    # 7. START APPLICATION AUDIO CAPTURE
    # ========================================================

    try:
        tap = ProcessAudioCapture(
            pid=target_pid,
            on_data=on_audio,
        )

        tap.start()

        print(f"Capturing PID {target_pid}")
        print("Press ESC to exit.")
        print("Audio diagnostics will appear every five seconds.")

        # ====================================================
        # 8. MAIN DISPLAY LOOP
        # ====================================================

        while running:

            # -----------------------------------------------
            # Keyboard and window events
            # -----------------------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    # Exit the visualizer.
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    # Toggle the information overlay.
                    elif event.key == pygame.K_h:
                        show_hud = not show_hud

                    # Recreate all particles.
                    elif event.key == pygame.K_r:
                        particles = [
                            new_particle(rng, width, height)
                            for _ in range(PARTICLE_COUNT)
                        ]

                    # Toggle fullscreen/windowed mode.
                    elif event.key == pygame.K_F11:

                        fullscreen = not fullscreen

                        if fullscreen:
                            screen = pygame.display.set_mode(
                                (0, 0), pygame.FULLSCREEN
                            )
                        else:
                            screen = pygame.display.set_mode(
                                (1000, 700), pygame.RESIZABLE
                            )

                        # Refresh dimensions after changing modes.
                        width, height = screen.get_size()
                        cx = width / 2
                        cy = height / 2

                # Support resizing the windowed display.
                elif event.type == pygame.VIDEORESIZE:
                    if not fullscreen:
                        width = max(200, event.w)
                        height = max(150, event.h)

                        screen = pygame.display.set_mode(
                            (width, height), pygame.RESIZABLE
                        )

                        cx = width / 2
                        cy = height / 2

            # -----------------------------------------------
            # Retrieve audio blocks from the callback queue
            # -----------------------------------------------

            try:
                while True:
                    block = audio_queue.get_nowait()

                    # Append new samples to those awaiting analysis.
                    pending = np.concatenate(
                        (pending, block), axis=0
                    )

            except queue.Empty:
                pass

            # -----------------------------------------------
            # Process audio in chunks
            # -----------------------------------------------

            while len(pending) >= 512:

                # Take one block of 512 stereo frames.
                block = pending[:512]
                pending = pending[512:]

                # Average left and right channels for mono analysis.
                mono = np.mean(block, axis=1)
                n = len(mono)

                # Shift the rolling buffer and append new samples.
                audio_buffer[:-n] = audio_buffer[n:]
                audio_buffer[-n:] = mono

                # Window the signal to reduce FFT edge artefacts.
                windowed = audio_buffer * np.hanning(
                    ANALYSIS_SIZE
                )

                # FFT converts the signal into frequency bins.
                spectrum = np.abs(
                    np.fft.rfft(windowed)
                )

                # Calculate the frequency corresponding to each bin.
                frequencies = np.fft.rfftfreq(
                    ANALYSIS_SIZE,
                    d=1.0 / SAMPLE_RATE,
                )

                # Select the bins in our chosen bass range.
                bass_bins = (
                    (frequencies >= BASS_MIN_HZ)
                    & (frequencies <= BASS_MAX_HZ)
                )

                # Measure bass energy.
                if np.any(bass_bins):
                    bass_level = float(
                        np.sqrt(
                            np.mean(
                                spectrum[bass_bins] ** 2
                            )
                        )
                    )

            # -----------------------------------------------
            # Smooth the bass measurement
            # -----------------------------------------------

            # Scale the measurement to the range 0.0 to 1.0.
            # Adjust 35.0 if your audio level is much higher/lower.
            normalised_bass = clamp(
                bass_level / 35.0, 0.0, 1.0
            )

            # A weighted average reduces sudden visual jumps.
            smoothed_bass = (
                0.82 * smoothed_bass
                + 0.18 * normalised_bass
            )

            # -----------------------------------------------
            # Calculate bass pulse
            # -----------------------------------------------

            # Stronger bass expands the central rings.
            pulse = 1.0 + smoothed_bass

            # Base size adapts to the screen dimensions.
            base_radius = min(width, height) * 0.115

            # Apply the bass pulse.
            ring_radius = base_radius * pulse

            # Clear the previous frame.
            screen.fill(BLACK)

            # -----------------------------------------------
            # Draw central rings
            # -----------------------------------------------

            pygame.draw.circle(
                screen,
                DIM_GREEN,
                (int(cx), int(cy)),
                max(2, int(ring_radius)),
                1,
            )

            pygame.draw.circle(
                screen,
                GREEN,
                (int(cx), int(cy)),
                max(2, int(ring_radius * 0.72)),
                1,
            )

            # -----------------------------------------------
            # Animate the particles
            # -----------------------------------------------

            current_time = time.monotonic()

            for particle in particles:

                # Move particles outward.
                # Bass increases their speed.
                particle["radius"] += (
                    particle["speed"]
                    * (0.35 + smoothed_bass * 2.4)
                    / FPS
                )

                # Add a subtle sideways oscillation.
                particle["angle"] += math.sin(
                    particle["phase"] + current_time * 0.5
                ) * 0.0015

                # Convert polar coordinates to screen coordinates.
                radius = particle["radius"]
                angle = particle["angle"]

                x = cx + math.cos(angle) * radius

                # Compress the vertical axis slightly for an oval look.
                y = cy + math.sin(angle) * radius * 0.72

                # Recycle particles that have moved too far away.
                if radius > max(width, height) * 0.72:
                    particle.update(
                        new_particle(rng, width, height)
                    )
                    continue

                # Larger particles become bright during strong bass.
                if (
                    smoothed_bass > 0.25
                    and particle["size"] >= 2
                ):
                    colour = GREEN
                else:
                    colour = DIM_GREEN

                # Draw this particle.
                pygame.draw.circle(
                    screen,
                    colour,
                    (int(x), int(y)),
                    particle["size"],
                )

            # -----------------------------------------------
            # Draw small orbiting dots
            # -----------------------------------------------

            # These dots follow the central pulse ring.
            for i in range(48):

                angle = (
                    i / 48.0 * math.tau
                    + current_time * 0.12
                )

                # Slight variation makes the orbit look alive.
                orbit = ring_radius * (
                    1.08
                    + 0.04 * math.sin(current_time * 2 + i)
                )

                x = cx + math.cos(angle) * orbit
                y = cy + math.sin(angle) * orbit * 0.72

                pygame.draw.circle(
                    screen,
                    GREEN,
                    (int(x), int(y)),
                    2,
                )

            # -----------------------------------------------
            # Audio diagnostics
            # -----------------------------------------------

            now = time.monotonic()

            if now - last_log_time >= LOG_INTERVAL:

                with info_lock:
                    info = latest_info["data"]

                if info is None:
                    print("[AUDIO] No callback data received.")

                else:
                    age = time.time() - info["time"]

                    print(
                        f"[AUDIO] age={age:.2f}s | "
                        f"callback_frames={info['callback_frames']} | "
                        f"bytes={info['bytes']} | "
                        f"float_values={info['float_values']} | "
                        f"decoded_frames={info['decoded_frames']} | "
                        f"peak={info['peak']:.6f} | "
                        f"RMS={info['rms']:.6f} | "
                        f"samples={info['first_samples']}"
                    )

                    # Nonzero bytes but zero samples may indicate
                    # silence OR an incorrect sample-format assumption.
                    if (
                        info["bytes"] > 0
                        and info["peak"] == 0.0
                    ):
                        print(
                            "[AUDIO] Decoded samples are zero. "
                            "Verify proc-tap's actual audio format."
                        )

                last_log_time = now

            # -----------------------------------------------
            # Optional on-screen information
            # -----------------------------------------------

            if show_hud:

                title = font.render(
                    "AUDIO-REACTIVE PARTICLES",
                    True,
                    GREEN,
                )

                subtitle = small_font.render(
                    f"APPLICATION AUDIO ONLY | PID {target_pid}",
                    True,
                    TEXT_GREEN,
                )

                controls = small_font.render(
                    "F11 fullscreen | H hide HUD | "
                    "R reset | ESC quit",
                    True,
                    DIM_GREEN,
                )

                meter_label = small_font.render(
                    f"BASS PULSE: {smoothed_bass:.2f}",
                    True,
                    GREEN,
                )

                screen.blit(title, (24, 20))
                screen.blit(subtitle, (25, 50))
                screen.blit(controls, (25, height - 34))

                # Background of the bass meter.
                pygame.draw.rect(
                    screen,
                    (25, 55, 30),
                    (25, 82, 130, 5),
                )

                # Filled portion shows the current bass level.
                pygame.draw.rect(
                    screen,
                    GREEN,
                    (
                        25,
                        82,
                        int(130 * smoothed_bass),
                        5,
                    ),
                )

                screen.blit(meter_label, (25, 94))

            # -----------------------------------------------
            # Display the finished frame
            # -----------------------------------------------

            # Present everything drawn in this frame.
            pygame.display.flip()

            # Limit CPU usage and target the chosen FPS.
            clock.tick(FPS)

    except KeyboardInterrupt:
        # Allow Ctrl+C in the terminal to stop the program.
        pass

    finally:
        # Stop capturing audio before closing Pygame.
        if tap is not None:
            try:
                tap.close()
            except Exception as exc:
                print(f"Capture cleanup warning: {exc}")

        pygame.quit()
        print("Visualizer stopped.")


# ============================================================
# 9. PROGRAM ENTRY POINT
# ============================================================

# Only run main() when this file is executed directly.
if __name__ == "__main__":
    main()
