
# ============================================================
# AUDIO-REACTIVE PARTICLE VISUALIZER
# ============================================================
# Features:
# - Captures audio from one application using proc-tap
# - No microphone input
# - Fullscreen on startup
# - F11: Toggle fullscreen/windowed
# - H: Toggle HUD
# - R: Reset audio analysis and particles
# - ESC: Quit
# - Bass-responsive concentric circles
# - Each circle ripples with a traveling sine wave
# - Treble-responsive sine wave across the screen
# - Green particles speed up with bass
# - Diagnostic logging every 5 seconds
#
# NOTE:
# The audio callback assumes proc-tap provides float32 PCM.
# If levels stay at zero while audio plays, verify the actual
# callback format before changing the sensitivity settings.
# ============================================================

import math
import time
import queue
import logging

import numpy as np
import pygame

from proctap import ProcessAudioCapture


# ============================================================
# CONFIGURATION
# ============================================================

# Audio capture
SAMPLE_RATE = 48000
CHANNELS = 2
AUDIO_QUEUE_SIZE = 64

# FFT analysis
ANALYSIS_SIZE = 4096

BASS_MIN_HZ = 40
BASS_MAX_HZ = 250

TREBLE_MIN_HZ = 2500
TREBLE_MAX_HZ = 10000

# ------------------------------------------------------------
# AUDIO SENSITIVITY
# These values are deliberately much lower than before.
# ------------------------------------------------------------

BASS_SENSITIVITY = 0.1
TREBLE_SENSITIVITY = 0.4

# Smoothing:
# Higher = slower and smoother response.
BASS_SMOOTHING = 0.82
TREBLE_SMOOTHING = 0.78

# Bass response curve:
# Higher = more compression of strong bass.
BASS_COMPRESSION = 1.8

# ------------------------------------------------------------
# VISUAL RESPONSE
# ------------------------------------------------------------

PULSE_STRENGTH = 1.15

# Bass rings
BASS_RING_COUNT = 6
BASS_RING_SPACING = 22
BASS_RING_BASE_RADIUS = 45

BASS_RING_WAVE_AMOUNT = 16.0
BASS_RING_WAVE_SPEED = 2.8
BASS_RING_WAVE_CYCLES = 3.0

# Treble wave
WAVE_AMPLITUDE = 145
WAVE_FREQUENCY = 2.5
WAVE_SPEED = 4.0

# Particles
PARTICLE_COUNT = 220
PARTICLE_BASS_SPEED = 7.0

# Display
FPS = 60
BACKGROUND_COLOR = (2, 6, 5)
GREEN = (0, 255, 100)
DIM_GREEN = (0, 100, 45)
WHITE = (210, 255, 225)

# Logging
LOG_INTERVAL = 5.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("AudioVisualiser")


# ============================================================
# AUDIO QUEUE AND DIAGNOSTICS
# ============================================================

audio_queue = queue.Queue(
    maxsize=AUDIO_QUEUE_SIZE
)

audio_diagnostics = {
    "callbacks": 0,
    "callback_frames": 0,
    "callback_bytes": 0,
    "last_peak": 0.0,
    "last_rms": 0.0,
    "last_error": None,
}


def on_audio(pcm, frames):
    """
    Receive audio from proc-tap.

    The callback copies incoming data into a queue so the
    capture thread doesn't need to run FFT calculations.
    """

    try:
        audio_diagnostics["callbacks"] += 1
        audio_diagnostics["callback_frames"] = frames
        audio_diagnostics["callback_bytes"] = len(pcm)

        # Assumption: proc-tap returns float32 PCM samples.
        samples = np.frombuffer(
            pcm,
            dtype=np.float32,
        ).copy()

        # Convert stereo audio to mono for analysis.
        if CHANNELS > 1:
            usable_length = (
                len(samples) // CHANNELS
            ) * CHANNELS

            samples = samples[:usable_length]

            if len(samples) > 0:
                samples = samples.reshape(
                    -1,
                    CHANNELS,
                ).mean(axis=1)

        # Diagnostics help identify silent or incorrectly
        # decoded audio.
        if len(samples) > 0:
            audio_diagnostics["last_peak"] = float(
                np.max(np.abs(samples))
            )

            audio_diagnostics["last_rms"] = float(
                np.sqrt(np.mean(samples ** 2))
            )

        # Add the audio block without blocking capture.
        try:
            audio_queue.put_nowait(samples)

        except queue.Full:
            # Drop the oldest block if the queue is full.
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                audio_queue.put_nowait(samples)
            except queue.Full:
                pass

    except Exception as exc:
        audio_diagnostics["last_error"] = str(exc)

        logger.exception(
            "Error in audio callback"
        )


# ============================================================
# AUDIO ANALYZER
# ============================================================

class AudioAnalyzer:
    """
    Maintains a rolling audio buffer and extracts bass and
    treble energy using an FFT.
    """

    def __init__(self):
        self.buffer = np.zeros(
            ANALYSIS_SIZE,
            dtype=np.float32,
        )

        self.window = np.hanning(
            ANALYSIS_SIZE
        ).astype(np.float32)

        self.frequencies = np.fft.rfftfreq(
            ANALYSIS_SIZE,
            d=1.0 / SAMPLE_RATE,
        )

        self.bass_mask = (
            (self.frequencies >= BASS_MIN_HZ)
            & (self.frequencies <= BASS_MAX_HZ)
        )

        self.treble_mask = (
            (self.frequencies >= TREBLE_MIN_HZ)
            & (self.frequencies <= TREBLE_MAX_HZ)
        )

        self.bass = 0.0
        self.treble = 0.0

        self.raw_bass = 0.0
        self.raw_treble = 0.0

    def update(self):
        """
        Read queued samples and update the audio levels.
        """

        new_blocks = []

        while True:
            try:
                block = audio_queue.get_nowait()

                if len(block) > 0:
                    new_blocks.append(block)

            except queue.Empty:
                break

        if new_blocks:
            incoming = np.concatenate(
                new_blocks
            )

            if len(incoming) >= ANALYSIS_SIZE:
                self.buffer[:] = incoming[-ANALYSIS_SIZE:]

            else:
                shift = len(incoming)

                self.buffer[:-shift] = self.buffer[shift:]
                self.buffer[-shift:] = incoming

        # Window the signal to reduce FFT edge effects.
        windowed = self.buffer * self.window

        spectrum = np.abs(
            np.fft.rfft(windowed)
        )

        if len(spectrum) == 0:
            return

        # Calculate mean energy in each frequency range.
        if np.any(self.bass_mask):
            self.raw_bass = float(
                np.mean(
                    spectrum[self.bass_mask]
                )
            )
        else:
            self.raw_bass = 0.0

        if np.any(self.treble_mask):
            self.raw_treble = float(
                np.mean(
                    spectrum[self.treble_mask]
                )
            )
        else:
            self.raw_treble = 0.0

        # ----------------------------------------------------
        # DAMPENED AUDIO RESPONSE
        # ----------------------------------------------------
        #
        # Unlike clipping amplified audio immediately at 1.0,
        # the exponential curve gives a gradual response.
        # Lower sensitivity means more headroom.
        # ----------------------------------------------------

        bass_scaled = (
            self.raw_bass * BASS_SENSITIVITY
        )

        treble_scaled = (
            self.raw_treble * TREBLE_SENSITIVITY
        )

        # Gradually approach maximum instead of abruptly
        # clipping the raw scaled values.
        bass_target = (
            1.0 - math.exp(-bass_scaled)
        )

        treble_target = (
            1.0 - math.exp(-treble_scaled)
        )

        # Additional bass compression.
        bass_target = (
            bass_target ** BASS_COMPRESSION
        )

        # Smooth bass transitions.
        self.bass += (
            bass_target - self.bass
        ) * (1.0 - BASS_SMOOTHING)

        # Smooth treble transitions.
        self.treble += (
            treble_target - self.treble
        ) * (1.0 - TREBLE_SMOOTHING)

        # Ensure levels remain within valid bounds.
        self.bass = float(
            np.clip(self.bass, 0.0, 1.0)
        )

        self.treble = float(
            np.clip(self.treble, 0.0, 1.0)
        )

    def reset(self):
        """Reset the analyzer to silence."""

        self.buffer.fill(0.0)

        self.bass = 0.0
        self.treble = 0.0

        self.raw_bass = 0.0
        self.raw_treble = 0.0


# ============================================================
# PARTICLES
# ============================================================

class Particle:
    """Green particle with bass-dependent movement."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.reset(
            random_position=True
        )

    def reset(self, random_position=False):
        self.angle = np.random.uniform(
            0.0,
            math.tau,
        )

        self.distance = np.random.uniform(
            20.0,
            min(self.width, self.height) * 0.5,
        )

        self.base_speed = np.random.uniform(
            0.15,
            0.8,
        )

        self.size = int(
            np.random.randint(1, 4)
        )

        self.alpha = int(
            np.random.randint(90, 256)
        )

        if random_position:
            self.distance = np.random.uniform(
                10.0,
                min(self.width, self.height) * 0.48,
            )

    def update(self, bass_level, dt):
        # The bass level increases particle speed.
        movement = (
            self.base_speed
            + bass_level * PARTICLE_BASS_SPEED
        )

        self.distance += (
            movement * dt * 30.0
        )

        # Particles rotate around the visualizer center.
        self.angle += (
            self.base_speed
            * dt
            * (1.0 + bass_level * 1.5)
        )

        maximum_distance = (
            min(self.width, self.height) * 0.52
        )

        # Respawn particles after they move off-screen.
        if self.distance > maximum_distance:
            self.distance = np.random.uniform(
                5.0,
                30.0,
            )

            self.angle = np.random.uniform(
                0.0,
                math.tau,
            )

    def draw(self, surface, center_x, center_y):
        x = (
            center_x
            + math.cos(self.angle) * self.distance
        )

        y = (
            center_y
            + math.sin(self.angle) * self.distance
        )

        color = (
            0,
            min(255, self.alpha),
            min(130, self.alpha // 2),
        )

        pygame.draw.circle(
            surface,
            color,
            (int(x), int(y)),
            self.size,
        )


# ============================================================
# BASS RINGS
# ============================================================

def draw_bass_rings(
    surface,
    center_x,
    center_y,
    base_radius,
    bass_level,
    animation_time,
):
    """
    Draw concentric bass rings.

    Every ring expands with the bass level and receives its
    own sine-wave ripple.
    """

    bass_level = float(
        np.clip(bass_level, 0.0, 1.0)
    )

    # Shared expansion, controlled by dampened bass.
    bass_expansion = (
        PULSE_STRENGTH * bass_level
    )

    for ring_index in range(
        BASS_RING_COUNT
    ):
        ring_radius = (
            base_radius
            + ring_index * BASS_RING_SPACING
        )

        # Offset each ring's wave phase.
        phase_offset = (
            ring_index * 0.65
        )

        # Stronger bass creates larger sine ripples.
        ripple_strength = (
            BASS_RING_WAVE_AMOUNT
            * bass_level
            * (1.0 + ring_index * 0.12)
        )

        points = []
        point_count = 180

        for i in range(point_count):
            angle = (
                i / point_count
            ) * math.tau

            ripple = math.sin(
                angle * BASS_RING_WAVE_CYCLES
                - animation_time * BASS_RING_WAVE_SPEED
                + phase_offset
            )

            # Overall expansion plus local sine displacement.
            radius = (
                ring_radius * (1.0 + bass_expansion)
                + ripple * ripple_strength
            )

            x = (
                center_x
                + math.cos(angle) * radius
            )

            y = (
                center_y
                + math.sin(angle) * radius
            )

            points.append(
                (int(x), int(y))
            )

        brightness = max(
            70,
            220 - ring_index * 22,
        )

        ring_color = (
            0,
            brightness,
            min(100, brightness // 2),
        )

        pygame.draw.lines(
            surface,
            ring_color,
            True,
            points,
            2 if ring_index < 3 else 1,
        )


# ============================================================
# TREBLE WAVE
# ============================================================

def draw_treble_wave(
    surface,
    center_y,
    screen_width,
    treble_level,
    animation_time,
):
    """Draw two treble-responsive sine waves."""

    treble_level = float(
        np.clip(treble_level, 0.0, 1.0)
    )

    # Primary wave amplitude.
    wave_height = (
        WAVE_AMPLITUDE * treble_level
    )

    # Small baseline movement keeps the wave visible at silence.
    wave_height += 2.0

    points = []
    point_spacing = 4

    for x in range(
        0,
        screen_width + point_spacing,
        point_spacing,
    ):
        phase = (
            (x / screen_width)
            * math.tau
            * WAVE_FREQUENCY
            + animation_time * WAVE_SPEED
        )

        y = (
            center_y
            + math.sin(phase) * wave_height
        )

        points.append(
            (x, int(y))
        )

    if len(points) >= 2:
        pygame.draw.lines(
            surface,
            GREEN,
            False,
            points,
            2,
        )

    # Secondary wave adds depth without overpowering the main.
    secondary_points = []

    secondary_height = (
        wave_height * 0.45
    )

    for x in range(
        0,
        screen_width + point_spacing,
        point_spacing,
    ):
        phase = (
            (x / screen_width)
            * math.tau
            * WAVE_FREQUENCY
            - animation_time * WAVE_SPEED * 0.7
            + 0.8
        )

        y = (
            center_y
            + math.sin(phase) * secondary_height
        )

        secondary_points.append(
            (x, int(y))
        )

    if len(secondary_points) >= 2:
        pygame.draw.lines(
            surface,
            DIM_GREEN,
            False,
            secondary_points,
            1,
        )


# ============================================================
# HUD
# ============================================================

def draw_meter(
    surface,
    x,
    y,
    width,
    height,
    value,
    label,
    font,
):
    """Draw an audio meter and its current normalized value."""

    value = float(
        np.clip(value, 0.0, 1.0)
    )

    label_surface = font.render(
        f"{label}: {value:.2f}",
        True,
        WHITE,
    )

    surface.blit(
        label_surface,
        (x, y - 22),
    )

    pygame.draw.rect(
        surface,
        (20, 45, 30),
        (x, y, width, height),
        border_radius=3,
    )

    fill_width = int(
        width * value
    )

    if fill_width > 0:
        pygame.draw.rect(
            surface,
            GREEN,
            (x, y, fill_width, height),
            border_radius=3,
        )


def draw_hud(
    surface,
    font,
    bass_level,
    treble_level,
    fullscreen,
):
    """Draw meters and keyboard controls."""

    draw_meter(
        surface,
        20,
        45,
        180,
        10,
        bass_level,
        "BASS",
        font,
    )

    draw_meter(
        surface,
        20,
        95,
        180,
        10,
        treble_level,
        "TREBLE",
        font,
    )

    mode = (
        "FULLSCREEN"
        if fullscreen
        else "WINDOWED"
    )

    help_text = (
        f"{mode} | F11: Toggle | H: HUD | "
        "R: Reset | ESC: Quit"
    )

    help_surface = font.render(
        help_text,
        True,
        DIM_GREEN,
    )

    surface.blit(
        help_surface,
        (20, 125),
    )


# ============================================================
# DISPLAY
# ============================================================

def create_display(fullscreen=True):
    """Create the fullscreen or windowed display."""

    if fullscreen:
        info = pygame.display.Info()

        return pygame.display.set_mode(
            (
                info.current_w,
                info.current_h,
            ),
            pygame.FULLSCREEN,
        )

    return pygame.display.set_mode(
        (1280, 720),
        pygame.RESIZABLE,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    pygame.init()

    pygame.display.set_caption(
        "Audio Reactive Visualizer"
    )

    # Start fullscreen.
    fullscreen = True

    screen = create_display(
        fullscreen=fullscreen
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        16,
    )

    width, height = screen.get_size()

    center_x = width // 2
    center_y = height // 2

    show_hud = True

    analyzer = AudioAnalyzer()

    particles = [
        Particle(width, height)
        for _ in range(PARTICLE_COUNT)
    ]

    # --------------------------------------------------------
    # TARGET PROCESS
    # --------------------------------------------------------
    # Enter the PID of the application whose audio you want
    # to capture. No microphone is used.
    #
    # To hard-code a PID, replace None with its integer:
    # TARGET_PID = 12345
    # --------------------------------------------------------

    TARGET_PID = None

    if TARGET_PID is None:
        # Temporarily close Pygame so the PID prompt is visible.
        pygame.quit()

        try:
            TARGET_PID = int(
                input(
                    "Enter the target application's PID: "
                ).strip()
            )

        except (ValueError, EOFError):
            print("Invalid PID. Exiting.")
            return

        # Reinitialize the fullscreen display.
        pygame.init()

        pygame.display.set_caption(
            "Audio Reactive Visualizer"
        )

        screen = create_display(
            fullscreen=fullscreen
        )

        clock = pygame.time.Clock()

        font = pygame.font.SysFont(
            "consolas",
            16,
        )

        width, height = screen.get_size()

        center_x = width // 2
        center_y = height // 2

        particles = [
            Particle(width, height)
            for _ in range(PARTICLE_COUNT)
        ]

    tap = None

    try:
        logger.info(
            "Starting process audio capture for PID %s",
            TARGET_PID,
        )

        tap = ProcessAudioCapture(
            pid=TARGET_PID,
            on_data=on_audio,
        )

        tap.start()

        logger.info(
            "Audio capture started. Play audio in the target app."
        )

        last_log_time = time.monotonic()

        running = True

        while running:
            dt = clock.tick(FPS) / 1000.0

            # Prevent huge updates after a display change.
            dt = min(dt, 0.05)

            # ------------------------------------------------
            # EVENTS
            # ------------------------------------------------

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_h:
                        show_hud = not show_hud

                    elif event.key == pygame.K_F11:
                        fullscreen = not fullscreen

                        screen = create_display(
                            fullscreen=fullscreen
                        )

                        width, height = screen.get_size()

                        center_x = width // 2
                        center_y = height // 2

                        particles = [
                            Particle(width, height)
                            for _ in range(PARTICLE_COUNT)
                        ]

                    elif event.key == pygame.K_r:
                        analyzer.reset()

                        # Clear queued stale audio as well.
                        while True:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break

                        for particle in particles:
                            particle.reset(
                                random_position=True
                            )

            # ------------------------------------------------
            # AUDIO
            # ------------------------------------------------

            analyzer.update()

            bass_level = analyzer.bass
            treble_level = analyzer.treble

            # ------------------------------------------------
            # PARTICLE UPDATE
            # ------------------------------------------------

            for particle in particles:
                particle.width = width
                particle.height = height

                particle.update(
                    bass_level,
                    dt,
                )

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            screen.fill(
                BACKGROUND_COLOR
            )

            animation_time = time.monotonic()

            # Concentric bass rings, each with sine ripples.
            draw_bass_rings(
                screen,
                center_x,
                center_y,
                base_radius=BASS_RING_BASE_RADIUS,
                bass_level=bass_level,
                animation_time=animation_time,
            )

            # Treble waveform.
            draw_treble_wave(
                screen,
                center_y,
                width,
                treble_level,
                animation_time,
            )

            # Bass-reactive particles.
            for particle in particles:
                particle.draw(
                    screen,
                    center_x,
                    center_y,
                )

            # Center dot expands with bass.
            center_radius = int(
                5 + bass_level * 12
            )

            pygame.draw.circle(
                screen,
                GREEN,
                (center_x, center_y),
                center_radius,
            )

            # HUD and audio meters.
            if show_hud:
                draw_hud(
                    screen,
                    font,
                    bass_level,
                    treble_level,
                    fullscreen,
                )

            pygame.display.flip()

            # ------------------------------------------------
            # DIAGNOSTICS
            # ------------------------------------------------

            now = time.monotonic()

            if now - last_log_time >= LOG_INTERVAL:
                logger.info(
                    "Audio diagnostics | "
                    "callbacks=%s | frames=%s | bytes=%s | "
                    "peak=%.6f | RMS=%.6f | "
                    "raw_bass=%.6f | raw_treble=%.6f | "
                    "bass=%.3f | treble=%.3f",
                    audio_diagnostics["callbacks"],
                    audio_diagnostics["callback_frames"],
                    audio_diagnostics["callback_bytes"],
                    audio_diagnostics["last_peak"],
                    audio_diagnostics["last_rms"],
                    analyzer.raw_bass,
                    analyzer.raw_treble,
                    bass_level,
                    treble_level,
                )

                if audio_diagnostics["last_error"]:
                    logger.error(
                        "Last callback error: %s",
                        audio_diagnostics["last_error"],
                    )

                last_log_time = now

    except Exception:
        logger.exception(
            "Visualizer encountered an error"
        )

    finally:
        if tap is not None:
            try:
                tap.close()
            except Exception:
                logger.exception(
                    "Error closing audio capture"
                )

        pygame.quit()

        logger.info(
            "Visualizer shut down."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
