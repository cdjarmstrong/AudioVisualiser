
"""
Audio-Reactive Visualizer
=========================
Platform: Windows 11
Libraries: pygame, numpy, proc-tap

Visuals:
- Bass: expanding circle in the centre.
- Treble: moving sine wave across the middle.
- Particles: accelerate as bass energy rises.
- Theme: minimal green on black.

Controls:
- F11: Toggle fullscreen/windowed
- H: Hide/show information overlay
- R: Reset particles
- ESC: Quit

Audio capture:
- Captures a selected application PID.
- Does not use the microphone.

IMPORTANT:
The callback decoder assumes float32 stereo at 48 kHz.
Verify proc-tap's actual callback format if diagnostics
show zero-valued samples during playback.
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
# 1. SETTINGS
# ============================================================

SAMPLE_RATE = 48000
CHANNELS = 2

AUDIO_QUEUE_SIZE = 64
LOG_INTERVAL = 5.0
FPS = 60

PARTICLE_COUNT = 220
ANALYSIS_SIZE = 4096
ANALYSIS_CHUNK = 512

# Frequency bands
BASS_MIN_HZ = 40
BASS_MAX_HZ = 250

TREBLE_MIN_HZ = 2500
TREBLE_MAX_HZ = 10000

# Bass circle
BASS_SENSITIVITY = 12.0
BASS_SMOOTHING = 0.65
PULSE_STRENGTH = 0.65

# Treble sine wave
TREBLE_SENSITIVITY = 15.0
TREBLE_SMOOTHING = 0.65
WAVE_AMPLITUDE = 100
WAVE_FREQUENCY = 2.5
WAVE_SPEED = 3.0

# Particle movement
PARTICLE_BASS_SPEED = 7.0

# Colours are RGB values.
BLACK = (0, 0, 0)
GREEN = (0, 255, 75)
DIM_GREEN = (0, 100, 35)
TEXT_GREEN = (170, 190, 170)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clamp(value, low, high):
    """Keep a value between a minimum and maximum."""
    return max(low, min(high, value))


def new_particle(rng, width, height):
    """Create a particle with random position and movement."""

    return {
        # Direction around the centre, in radians.
        "angle": float(rng.uniform(0, math.tau)),

        # Distance from the centre.
        "radius": float(
            rng.uniform(20, min(width, height) * 0.45)
        ),

        # Outward movement speed.
        "speed": float(rng.uniform(15, 100)),

        # Particle size in pixels.
        "size": int(rng.choice([1, 1, 1, 2, 2, 3])),

        # Random phase for subtle movement.
        "phase": float(rng.uniform(0, math.tau)),
    }


# ============================================================
# 3. MAIN PROGRAM
# ============================================================

def main():

    if platform.system() != "Windows":
        raise RuntimeError(
            "This program is intended for Windows 11."
        )

    print("=" * 55)
    print("BASS CIRCLE + TREBLE SINE WAVE VISUALIZER")
    print("=" * 55)
    print("Application audio only. No microphone.")
    print("Find the application PID in Task Manager > Details.")
    print()

    try:
        target_pid = int(
            input("Enter target application PID: ").strip()
        )
    except ValueError:
        raise SystemExit("Please enter a valid numeric PID.")

    if target_pid <= 0:
        raise SystemExit("PID must be positive.")

    # ========================================================
    # 4. THREAD-SAFE AUDIO QUEUE AND DIAGNOSTICS
    # ========================================================

    # Audio arrives in the capture callback and is processed
    # separately by the Pygame main loop.
    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

    latest_info = {"data": None}
    info_lock = threading.Lock()

    # ========================================================
    # 5. AUDIO CALLBACK
    # ========================================================

    def on_audio(pcm, frames):
        """
        Called by proc-tap for each captured audio block.

        Assumes pcm contains float32 stereo samples.
        The actual callback format must be verified if needed.
        """

        try:
            byte_count = len(pcm)

            # Convert raw bytes into floating-point samples.
            samples = np.frombuffer(
                pcm, dtype=np.float32
            ).copy()

            # Keep complete stereo frames only.
            usable = (
                samples.size // CHANNELS
            ) * CHANNELS

            if usable == 0:
                return

            # Shape is now: (number_of_frames, 2).
            samples = samples[:usable].reshape(-1, CHANNELS)

            # Calculate signal diagnostics.
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(samples ** 2)))

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

            # Keep the callback responsive.
            try:
                audio_queue.put_nowait(samples)

            except queue.Full:
                # Discard an old block if processing falls behind.
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    audio_queue.put_nowait(samples)
                except queue.Full:
                    pass

        except (ValueError, TypeError, BufferError) as exc:
            print(f"[AUDIO] Decode error: {exc}")

    # ========================================================
    # 6. INITIALISE PYGAME AND PARTICLES
    # ========================================================

    pygame.init()
    pygame.display.set_caption(
        "Bass Circle + Treble Wave - App Audio Only"
    )

    # Start in fullscreen mode.
    display_info = pygame.display.Info()

    screen = pygame.display.set_mode(
        (display_info.current_w, display_info.current_h),
        pygame.FULLSCREEN,
    )

    width, height = screen.get_size()
    cx, cy = width / 2, height / 2

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 27)
    small_font = pygame.font.Font(None, 20)

    rng = np.random.default_rng()

    particles = [
        new_particle(rng, width, height)
        for _ in range(PARTICLE_COUNT)
    ]

    # ========================================================
    # 7. AUDIO ANALYSIS STATE
    # ========================================================

    # Rolling buffer of the most recent mono audio samples.
    audio_buffer = np.zeros(
        ANALYSIS_SIZE, dtype=np.float32
    )

    # Stores audio until a complete analysis chunk is available.
    pending = np.empty(
        (0, CHANNELS), dtype=np.float32
    )

    # Raw audio energy for each frequency band.
    bass_level = 0.0
    treble_level = 0.0

    # Smoothed values drive the animation.
    smoothed_bass = 0.0
    smoothed_treble = 0.0

    # Runtime state.
    running = True
    fullscreen = True
    show_hud = True
    tap = None

    last_log_time = time.monotonic()

    # ========================================================
    # 8. START APPLICATION AUDIO CAPTURE
    # ========================================================

    try:
        tap = ProcessAudioCapture(
            pid=target_pid,
            on_data=on_audio,
        )
        tap.start()

        print(f"Capturing application PID {target_pid}")
        print("F11: fullscreen | H: hide HUD | R: reset | ESC: quit")

        # ====================================================
        # 9. MAIN LOOP
        # ====================================================

        while running:

            # ------------------------------------------------
            # Keyboard and window events
            # ------------------------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_h:
                        show_hud = not show_hud

                    elif event.key == pygame.K_r:
                        particles = [
                            new_particle(rng, width, height)
                            for _ in range(PARTICLE_COUNT)
                        ]

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

                        width, height = screen.get_size()
                        cx, cy = width / 2, height / 2

                elif event.type == pygame.VIDEORESIZE:
                    if not fullscreen:
                        width = max(200, event.w)
                        height = max(150, event.h)

                        screen = pygame.display.set_mode(
                            (width, height), pygame.RESIZABLE
                        )

                        cx, cy = width / 2, height / 2

            # ------------------------------------------------
            # Collect incoming audio blocks
            # ------------------------------------------------

            try:
                while True:
                    block = audio_queue.get_nowait()

                    pending = np.concatenate(
                        (pending, block), axis=0
                    )

            except queue.Empty:
                pass

            # ------------------------------------------------
            # FFT ANALYSIS: BASS AND TREBLE
            # ------------------------------------------------

            while len(pending) >= ANALYSIS_CHUNK:

                # Extract one chunk of stereo audio.
                block = pending[:ANALYSIS_CHUNK]
                pending = pending[ANALYSIS_CHUNK:]

                # Average the left and right channels to mono.
                mono = np.mean(block, axis=1)
                n = len(mono)

                # Shift older samples out of the rolling buffer.
                audio_buffer[:-n] = audio_buffer[n:]

                # Append the latest samples.
                audio_buffer[-n:] = mono

                # Reduce FFT edge artefacts with a Hann window.
                windowed = audio_buffer * np.hanning(
                    ANALYSIS_SIZE
                )

                # Transform time-domain audio into frequency bins.
                spectrum = np.abs(
                    np.fft.rfft(windowed)
                )

                frequencies = np.fft.rfftfreq(
                    ANALYSIS_SIZE,
                    d=1.0 / SAMPLE_RATE,
                )

                # Select the bass frequency bins.
                bass_bins = (
                    (frequencies >= BASS_MIN_HZ)
                    & (frequencies <= BASS_MAX_HZ)
                )

                # Select the treble frequency bins.
                treble_bins = (
                    (frequencies >= TREBLE_MIN_HZ)
                    & (frequencies <= TREBLE_MAX_HZ)
                )

                # Calculate bass energy.
                if np.any(bass_bins):
                    bass_level = float(
                        np.sqrt(
                            np.mean(spectrum[bass_bins] ** 2)
                        )
                    )

                # Calculate treble energy.
                if np.any(treble_bins):
                    treble_level = float(
                        np.sqrt(
                            np.mean(spectrum[treble_bins] ** 2)
                        )
                    )

            # ------------------------------------------------
            # SMOOTH THE BASS RESPONSE
            # ------------------------------------------------

            # Convert bass energy into a convenient 0-to-1 range.
            normalised_bass = clamp(
                bass_level / BASS_SENSITIVITY,
                0.0,
                1.0,
            )

            # A lower smoothing factor reacts faster to beats.
            smoothed_bass = (
                BASS_SMOOTHING * smoothed_bass
                + (1.0 - BASS_SMOOTHING) * normalised_bass
            )

            # ------------------------------------------------
            # SMOOTH THE TREBLE RESPONSE
            # ------------------------------------------------

            normalised_treble = clamp(
                treble_level / TREBLE_SENSITIVITY,
                0.0,
                1.0,
            )

            smoothed_treble = (
                TREBLE_SMOOTHING * smoothed_treble
                + (1.0 - TREBLE_SMOOTHING) * normalised_treble
            )

            # ------------------------------------------------
            # DRAW THE BACKGROUND AND BASS CIRCLE
            # ------------------------------------------------

            screen.fill(BLACK)

            # Bass expands the circle.
            pulse = 1.0 + PULSE_STRENGTH * smoothed_bass

            base_radius = min(width, height) * 0.115
            ring_radius = base_radius * pulse

            # Outer ring.
            pygame.draw.circle(
                screen,
                DIM_GREEN,
                (int(cx), int(cy)),
                max(2, int(ring_radius)),
                2,
            )

            # Inner ring.
            pygame.draw.circle(
                screen,
                GREEN,
                (int(cx), int(cy)),
                max(2, int(ring_radius * 0.72)),
                1,
            )

            # Smaller centre ring reacts subtly to bass.
            pygame.draw.circle(
                screen,
                DIM_GREEN,
                (int(cx), int(cy)),
                max(2, int(ring_radius * 0.45)),
                1,
            )

            # ------------------------------------------------
            # DRAW THE TREBLE SINE WAVE ACROSS THE MIDDLE
            # ------------------------------------------------

            wave_points = []

            # Treble energy controls the wave's height.
            amplitude = WAVE_AMPLITUDE * smoothed_treble

            # Continuous time-based phase makes the wave travel.
            phase = time.monotonic() * WAVE_SPEED

            # More cycles = more waves across the screen.
            for x in range(0, width, 4):

                position = (
                    (x / width)
                    * math.tau
                    * WAVE_FREQUENCY
                )

                # Sine controls the vertical displacement.
                y = cy + math.sin(position + phase) * amplitude

                wave_points.append((x, int(y)))

            if len(wave_points) >= 2:
                pygame.draw.lines(
                    screen,
                    GREEN,
                    False,
                    wave_points,
                    2,
                )

                # Draw a dimmer second line for a subtle glow effect.
                glow_points = [
                    (x, y + 3) for x, y in wave_points
                ]

                pygame.draw.lines(
                    screen,
                    DIM_GREEN,
                    False,
                    glow_points,
                    1,
                )

            # ------------------------------------------------
            # PARTICLE ANIMATION
            # ------------------------------------------------

            current_time = time.monotonic()

            for particle in particles:

                # Stronger bass pushes particles outward faster.
                particle["radius"] += (
                    particle["speed"]
                    * (
                        0.5
                        + smoothed_bass * PARTICLE_BASS_SPEED
                    )
                    / FPS
                )

                # Add slight angular movement.
                particle["angle"] += math.sin(
                    particle["phase"] + current_time * 0.5
                ) * 0.0015

                radius = particle["radius"]
                angle = particle["angle"]

                # Convert polar coordinates to screen coordinates.
                x = cx + math.cos(angle) * radius
                y = (
                    cy
                    + math.sin(angle) * radius * 0.72
                )

                # Respawn particles that travel too far away.
                if radius > max(width, height) * 0.72:
                    particle.update(
                        new_particle(rng, width, height)
                    )
                    continue

                # Brighten particles when bass is detected.
                if (
                    smoothed_bass > 0.10
                    and particle["size"] >= 2
                ):
                    colour = GREEN
                else:
                    colour = DIM_GREEN

                pygame.draw.circle(
                    screen,
                    colour,
                    (int(x), int(y)),
                    particle["size"],
                )

            # ------------------------------------------------
            # ORBITING DOTS AROUND THE BASS CIRCLE
            # ------------------------------------------------

            for i in range(48):

                angle = (
                    i / 48.0 * math.tau
                    + current_time * 0.12
                )

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

            # ------------------------------------------------
            # PERIODIC AUDIO DIAGNOSTICS
            # ------------------------------------------------

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
                        f"RMS={info['rms']:.6f}"
                    )

                    print(
                        f"        bass={smoothed_bass:.3f} | "
                        f"treble={smoothed_treble:.3f}"
                    )

                    if (
                        info["bytes"] > 0
                        and info["peak"] == 0.0
                    ):
                        print(
                            "[AUDIO] Samples decode as zero. "
                            "Verify proc-tap's actual PCM format."
                        )

                last_log_time = now

            # ------------------------------------------------
            # OPTIONAL INFORMATION OVERLAY
            # ------------------------------------------------

            if show_hud:

                title = font.render(
                    "BASS CIRCLE + TREBLE WAVE",
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
                    "R reset particles | ESC quit",
                    True,
                    DIM_GREEN,
                )

                bass_text = small_font.render(
                    f"BASS: {smoothed_bass:.2f}",
                    True,
                    GREEN,
                )

                treble_text = small_font.render(
                    f"TREBLE: {smoothed_treble:.2f}",
                    True,
                    GREEN,
                )

                screen.blit(title, (24, 20))
                screen.blit(subtitle, (25, 50))
                screen.blit(controls, (25, height - 34))

                screen.blit(bass_text, (25, 82))
                screen.blit(treble_text, (25, 105))

                # Bass meter.
                pygame.draw.rect(
                    screen,
                    (25, 55, 30),
                    (25, 132, 130, 5),
                )

                pygame.draw.rect(
                    screen,
                    GREEN,
                    (
                        25,
                        132,
                        int(130 * smoothed_bass),
                        5,
                    ),
                )

                # Treble meter.
                pygame.draw.rect(
                    screen,
                    (25, 55, 30),
                    (25, 145, 130, 5),
                )

                pygame.draw.rect(
                    screen,
                    GREEN,
                    (
                        25,
                        145,
                        int(130 * smoothed_treble),
                        5,
                    ),
                )

            # ------------------------------------------------
            # PRESENT THE FRAME
            # ------------------------------------------------

            pygame.display.flip()
            clock.tick(FPS)

    except KeyboardInterrupt:
        # Support Ctrl+C in the terminal.
        pass

    finally:
        # Stop the capture before shutting down Pygame.
        if tap is not None:
            try:
                tap.close()
            except Exception as exc:
                print(f"Capture cleanup warning: {exc}")

        pygame.quit()
        print("Visualizer stopped.")


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
