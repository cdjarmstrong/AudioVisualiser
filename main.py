
import math
import platform
import queue
import threading
import time

import numpy as np
import pygame
from proctap import ProcessAudioCapture

SAMPLE_RATE = 48000
CHANNELS = 2
AUDIO_QUEUE_SIZE = 64
LOG_INTERVAL = 5.0
PARTICLE_COUNT = 220
FPS = 60

BLACK = (0, 0, 0)
GREEN = (0, 255, 75)
DIM_GREEN = (0, 100, 35)


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    if platform.system() != "Windows":
        raise RuntimeError("This version is intended for Windows 11.")

    print("Audio-reactive particle visualizer")
    print("Application audio only — no microphone")
    print("Find the target PID in Task Manager > Details.")

    try:
        target_pid = int(input("Enter target application PID: ").strip())
    except ValueError:
        raise SystemExit("Please enter a valid numeric PID.")

    if target_pid <= 0:
        raise SystemExit("PID must be positive.")

    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
    info_lock = threading.Lock()
    latest_info = {"data": None}

    def on_audio(pcm, frames):
        # This decoder assumes float32 stereo audio.
        # Verify proc-tap's actual callback format if readings are zero.
        try:
            byte_count = len(pcm)
            samples = np.frombuffer(pcm, dtype=np.float32).copy()

            usable = (samples.size // CHANNELS) * CHANNELS
            if usable == 0:
                return

            samples = samples[:usable].reshape(-1, CHANNELS)
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
                "first_samples": samples.flatten()[:10].tolist(),
            }

            with info_lock:
                latest_info["data"] = info

            try:
                audio_queue.put_nowait(samples)
            except queue.Full:
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

    pygame.init()
    pygame.display.set_caption(
        "Arkam's Audio Visualizer - Application Audio Only"
    )

    display_info = pygame.display.Info()
    screen = pygame.display.set_mode(
        (display_info.current_w, display_info.current_h),
        pygame.FULLSCREEN,
    )
    width, height = screen.get_size()

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 27)
    small_font = pygame.font.Font(None, 20)

    rng = np.random.default_rng()
    cx, cy = width / 2, height / 2

    def new_particle():
        return {
            "angle": float(rng.uniform(0, math.tau)),
            "radius": float(
                rng.uniform(20, min(width, height) * 0.45)
            ),
            "speed": float(rng.uniform(15, 100)),
            "size": int(rng.choice([1, 1, 1, 2, 2, 3])),
            "phase": float(rng.uniform(0, math.tau)),
        }

    particles = [new_particle() for _ in range(PARTICLE_COUNT)]

    # Rolling audio buffer for low-frequency analysis.
    analysis_size = 4096
    audio_buffer = np.zeros(analysis_size, dtype=np.float32)
    pending = np.empty((0, CHANNELS), dtype=np.float32)

    bass_level = 0.0
    smoothed_bass = 0.0
    last_log_time = time.monotonic()
    running = True
    fullscreen = True
    show_hud = True
    tap = None

    try:
        tap = ProcessAudioCapture(
            pid=target_pid,
            on_data=on_audio,
        )
        tap.start()

        print(f"Capturing application PID {target_pid}")
        print("F11: fullscreen | H: hide HUD | R: reset | Esc: quit")

        while running:
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
                            new_particle()
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

            # Gather callback audio blocks.
            try:
                while True:
                    block = audio_queue.get_nowait()
                    pending = np.concatenate(
                        (pending, block), axis=0
                    )
            except queue.Empty:
                pass

            # Update the audio analysis buffer.
            while len(pending) >= 512:
                block = pending[:512]
                pending = pending[512:]

                mono = np.mean(block, axis=1)
                n = len(mono)

                audio_buffer[:-n] = audio_buffer[n:]
                audio_buffer[-n:] = mono

                windowed = audio_buffer * np.hanning(
                    analysis_size
                )
                spectrum = np.abs(np.fft.rfft(windowed))
                frequencies = np.fft.rfftfreq(
                    analysis_size, d=1.0 / SAMPLE_RATE
                )

                bass_bins = (
                    (frequencies >= 40)
                    & (frequencies <= 250)
                )

                if np.any(bass_bins):
                    bass_level = float(
                        np.sqrt(np.mean(spectrum[bass_bins] ** 2))
                    )

            # Smooth the bass response to avoid jitter.
            normalised_bass = clamp(bass_level / 35.0, 0.0, 1.0)
            smoothed_bass = (
                0.82 * smoothed_bass
                + 0.18 * normalised_bass
            )

            pulse = 1.0 + 0.22 * smoothed_bass
            base_radius = min(width, height) * 0.115
            ring_radius = base_radius * pulse

            screen.fill(BLACK)

            # Bass-reactive central rings.
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

            current_time = time.monotonic()

            # Particle field.
            for particle in particles:
                particle["radius"] += (
                    particle["speed"]
                    * (0.35 + smoothed_bass * 2.4)
                    / FPS
                )

                particle["angle"] += math.sin(
                    particle["phase"] + current_time * 0.5
                ) * 0.0015

                radius = particle["radius"]
                angle = particle["angle"]

                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius * 0.72

                if radius > max(width, height) * 0.72:
                    particle.update(new_particle())
                    continue

                size = particle["size"]
                color = (
                    GREEN
                    if smoothed_bass > 0.25 and size >= 2
                    else DIM_GREEN
                )

                pygame.draw.circle(
                    screen,
                    color,
                    (int(x), int(y)),
                    size,
                )

            # Bright dots orbiting the pulse ring.
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

            # Print diagnostics every five seconds.
            now = time.monotonic()

            if now - last_log_time >= LOG_INTERVAL:
                with info_lock:
                    audio_info = latest_info["data"]

                if audio_info is None:
                    print("[AUDIO] No callback data received.")
                else:
                    age = time.time() - audio_info["time"]

                    print(
                        f"[AUDIO] age={age:.2f}s | "
                        f"callback frames="
                        f"{audio_info['callback_frames']} | "
                        f"bytes={audio_info['bytes']} | "
                        f"float32 values="
                        f"{audio_info['float_values']} | "
                        f"decoded frames="
                        f"{audio_info['decoded_frames']} | "
                        f"peak={audio_info['peak']:.6f} | "
                        f"RMS={audio_info['rms']:.6f} | "
                        f"samples={audio_info['first_samples']}"
                    )

                    if (
                        audio_info["bytes"] > 0
                        and audio_info["peak"] == 0.0
                    ):
                        print(
                            "[AUDIO] Samples decode as zero. "
                            "Check proc-tap's PCM format."
                        )

                last_log_time = now

            # Minimal green HUD.
            if show_hud:
                title = font.render(
                    "ARKAM'S AUDIO VISUALIZER", True, GREEN
                )
                subtitle = small_font.render(
                    f"APPLICATION AUDIO ONLY | PID {target_pid} "
                    "| NO MICROPHONE",
                    True,
                    (170, 190, 170),
                )
                controls = small_font.render(
                    "F11 fullscreen | H hide HUD | "
                    "R reset particles | Esc quit",
                    True,
                    (120, 150, 120),
                )
                meter_label = small_font.render(
                    f"BASS PULSE  {smoothed_bass:.2f}",
                    True,
                    GREEN,
                )

                screen.blit(title, (24, 20))
                screen.blit(subtitle, (25, 50))
                screen.blit(controls, (25, height - 34))

                pygame.draw.rect(
                    screen, (25, 55, 30),
                    (25, 82, 130, 5)
                )
                pygame.draw.rect(
                    screen, GREEN,
                    (25, 82, int(130 * smoothed_bass), 5)
                )
                screen.blit(meter_label, (25, 94))

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        if tap is not None:
            try:
                tap.close()
            except Exception as exc:
                print(f"Capture cleanup warning: {exc}")

        pygame.quit()


if __name__ == "__main__":
    main()
