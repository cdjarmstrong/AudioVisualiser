
# ============================================================
# AUDIO-REACTIVE PARTICLE VISUALIZER
# ------------------------------------------------------------
# Features:
# - Captures audio from one application using proc-tap
# - No microphone input
# - Fullscreen on startup
# - F11: Toggle fullscreen/windowed
# - H: Toggle HUD
# - R: Reset visual effects
# - ESC: Quit
# - Bass-responsive concentric circles
# - Bass sine ripples affect every circle
# - Treble-responsive sine wave
# - Green particles react to bass intensity
# - Audio diagnostics printed every 5 seconds
#
# IMPORTANT:
# This version assumes proc-tap delivers float32 PCM audio.
# Confirm the callback format if the diagnostic levels remain
# at zero despite the target application playing audio.
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

# Audio settings
SAMPLE_RATE = 48000
CHANNELS = 2
AUDIO_QUEUE_SIZE = 64

# Analysis settings
ANALYSIS_SIZE = 4096
ANALYSIS_CHUNK = 512

BASS_MIN_HZ = 40
BASS_MAX_HZ = 250

TREBLE_MIN_HZ = 2500
TREBLE_MAX_HZ = 10000

# Bass response
BASS_SENSITIVITY = 12.0
BASS_SMOOTHING = 0.72

# Stronger circle expansion, with compressed input response
PULSE_STRENGTH = 1.15

# Ring ripple settings
BASS_RING_COUNT = 6
BASS_RING_SPACING = 22
BASS_RING_BASE_RADIUS = 45

BASS_RING_WAVE_AMOUNT = 16.0
BASS_RING_WAVE_SPEED = 2.8
BASS_RING_WAVE_CYCLES = 3.0

# Treble response
TREBLE_SENSITIVITY = 22.0
TREBLE_SMOOTHING = 0.55

WAVE_AMPLITUDE = 145
WAVE_FREQUENCY = 2.5
WAVE_SPEED = 4.0

# Particle settings
PARTICLE_COUNT = 220
PARTICLE_BASS_SPEED = 7.0

# Display settings
FPS = 60
BACKGROUND_COLOR = (2, 6, 5)
GREEN = (0, 255, 100)
DIM_GREEN = (0, 100, 45)
WHITE = (210, 255, 225)

# Diagnostic logging interval
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

audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

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
    proc-tap audio callback.

    This callback copies the incoming PCM data into a queue so
    the audio-capture thread does not perform FFT calculations
    or drawing operations.

    Assumption: incoming PCM data is float32.
    """

    try:
        audio_diagnostics["callbacks"] += 1
        audio_diagnostics["callback_frames"] = frames
        audio_diagnostics["callback_bytes"] = len(pcm)

        # Decode the incoming PCM buffer as float32.
        samples = np.frombuffer(
            pcm,
            dtype=np.float32,
        ).copy()

        # Convert stereo data into a single mono signal for FFT.
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

        # Collect useful diagnostics.
        if len(samples) > 0:
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(samples ** 2)))

            audio_diagnostics["last_peak"] = peak
            audio_diagnostics["last_rms"] = rms

        # Queue the samples without blocking the callback.
        try:
            audio_queue.put_nowait(samples)
        except queue.Full:
            # Drop the oldest audio block to avoid stale audio.
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
        logger.exception("Error in audio callback")


# ============================================================
# AUDIO ANALYZER
# ============================================================

class AudioAnalyzer:
    """
    Maintains a rolling audio buffer and calculates bass/treble
    energy using an FFT.
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
        Consume available queued audio blocks, then update
        the frequency analysis.
        """

        new_samples = []

        while True:
            try:
                block = audio_queue.get_nowait()
                if len(block):
                    new_samples.append(block)
            except queue.Empty:
                break

        if new_samples:
            incoming = np.concatenate(new_samples)

            if len(incoming) >= ANALYSIS_SIZE:
                self.buffer[:] = incoming[-ANALYSIS_SIZE:]
            else:
                shift = len(incoming)

                self.buffer[:-shift] = self.buffer[shift:]
                self.buffer[-shift:] = incoming

        # If no audio is arriving, the previous buffer is retained.
        # A truly silent input will naturally produce near-zero FFT.
        windowed = self.buffer * self.window

        spectrum = np.abs(
            np.fft.rfft(windowed)
        )

        if len(spectrum) == 0:
            return

        # Mean energy within each frequency range.
        if np.any(self.bass_mask):
            self.raw_bass = float(
                np.mean(spectrum[self.bass_mask])
            )
        else:
            self.raw_bass = 0.0

        if np.any(self.treble_mask):
            self.raw_treble = float(
                np.mean(spectrum[self.treble_mask])
            )
        else:
            self.raw_treble = 0.0

        # Scale FFT values into more useful ranges.
        bass_scaled = min(
            1.0,
            self.raw_bass * BASS_SENSITIVITY,
        )

        treble_scaled = min(
            1.0,
            self.raw_treble * TREBLE_SENSITIVITY,
        )

        # Compress bass response so normal bass doesn't
        # immediately reach full intensity.
        bass_target = bass_scaled ** 1.8

        # Smooth the response.
        self.bass += (
            bass_target - self.bass
        ) * (1.0 - BASS_SMOOTHING)

        self.treble += (
            treble_scaled - self.treble
        ) * (1.0 - TREBLE_SMOOTHING)

        self.bass = float(
            np.clip(self.bass, 0.0, 1.0)
        )

        self.treble = float(
            np.clip(self.treble, 0.0, 1.0)
        )

    def reset(self):
        """Clear the audio analysis state."""

        self.buffer.fill(0.0)

        self.bass = 0.0
        self.treble = 0.0

        self.raw_bass = 0.0
        self.raw_treble = 0.0


# ============================================================
# PARTICLES
# ============================================================

class Particle:
    """
    A green particle that moves around the center and is
    influenced by the current bass level.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.reset(random_position=True)

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

        self.size = np.random.randint(1, 4)

        self.alpha = np.random.randint(
            90,
            256,
        )

        if random_position:
            self.distance = np.random.uniform(
                10.0,
                min(self.width, self.height) * 0.48,
            )

    def update(self, bass_level, dt):
        # Bass increases the outward movement speed.
        movement = (
            self.base_speed
            + bass_level * PARTICLE_BASS_SPEED
        )

        self.distance += movement * dt * 30.0

        # Rotate particles around the center.
        self.angle += (
            self.base_speed * dt
            * (1.0 + bass_level * 1.5)
        )

        maximum_distance = (
            min(self.width, self.height) * 0.52
        )

        # Respawn particles once they leave the visual area.
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
        x = center_x + math.cos(
            self.angle
        ) * self.distance

        y = center_y + math.sin(
            self.angle
        ) * self.distance

        # Brighter particles are drawn slightly larger.
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
# BASS RING DRAWING
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
    Draw concentric rings that expand with bass.

    Every ring receives its own traveling sine ripple.
    Different phase offsets make the rings move independently
    while remaining synchronized to the bass intensity.
    """

    bass_level = float(
        np.clip(bass_level, 0.0, 1.0)
    )

    # Overall expansion shared by all rings.
    bass_expansion = (
        PULSE_STRENGTH * bass_level
    )

    for ring_index in range(BASS_RING_COUNT):
        # Each ring has a different base radius.
        ring_radius = (
            base_radius
            + ring_index * BASS_RING_SPACING
        )

        # Offset the ripple phase for each concentric ring.
        phase_offset = ring_index * 0.65

        # Bass increases the strength of the sine distortion.
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

            # Traveling sine wave around the circle.
            ripple = math.sin(
                angle * BASS_RING_WAVE_CYCLES
                - animation_time * BASS_RING_WAVE_SPEED
                + phase_offset
            )

            # Apply the overall bass expansion and local ripple.
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

        # Fade outer rings slightly.
        brightness = max(
            70,
            220 - ring_index * 22,
        )

        ring_color = (
            0,
            brightness,
            min(100, brightness // 2),
        )

        if len(points) >= 2:
            pygame.draw.lines(
                surface,
                ring_color,
                True,
                points,
                2 if ring_index < 3 else 1,
            )


# ============================================================
# TREBLE WAVE DRAWING
# ============================================================

def draw_treble_wave(
    surface,
    center_y,
    screen_width,
    treble_level,
    animation_time,
):
    """
    Draw a sine wave across the screen.

    The treble level controls the wave height.
    """

    treble_level = float(
        np.clip(treble_level, 0.0, 1.0)
    )

    wave_height = (
        WAVE_AMPLITUDE * treble_level
    )

    # Keep a small visible wave even when treble is quiet.
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

    # Draw a softer secondary wave for extra depth.
    secondary_points = []

    secondary_height = wave_height * 0.45

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
    """Draw a labeled horizontal audio-level meter."""

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
    """Draw the audio meters and keyboard controls."""

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
# DISPLAY MANAGEMENT
# ============================================================

def create_display(fullscreen=True):
    """Create either a fullscreen or windowed display."""

    if fullscreen:
        info = pygame.display.Info()

        screen = pygame.display.set_mode(
            (
                info.current_w,
                info.current_h,
            ),
            pygame.FULLSCREEN,
        )
    else:
        screen = pygame.display.set_mode(
            (1280, 720),
            pygame.RESIZABLE,
        )

    return screen


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    pygame.init()

    pygame.display.set_caption(
        "Audio Reactive Visualizer"
    )

    # Start in fullscreen mode.
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

    # Audio analysis object.
    analyzer = AudioAnalyzer()

    # Create particles.
    particles = [
        Particle(width, height)
        for _ in range(PARTICLE_COUNT)
    ]

    # --------------------------------------------------------
    # Choose the target application's process ID.
    # --------------------------------------------------------
    #
    # Set TARGET_PID to the PID of the application you want
    # to capture. Do not use the visualizer's own PID.
    #
    # Example:
    # TARGET_PID = 12345
    #
    # If TARGET_PID is None, the program asks for it at launch.
    # --------------------------------------------------------

    TARGET_PID = None

    if TARGET_PID is None:
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

    # Start capturing audio from the chosen process.
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
            "Audio capture started. "
            "Make sure the target application is playing audio."
        )

        last_log_time = time.monotonic()

        running = True

        while running:
            dt = clock.tick(FPS) / 1000.0

            # Limit very large time steps after window changes.
            dt = min(dt, 0.05)

            # ------------------------------------------------
            # HANDLE EVENTS
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

                        for particle in particles:
                            particle.reset(
                                random_position=True
                            )

            # ------------------------------------------------
            # UPDATE AUDIO ANALYSIS
            # ------------------------------------------------

            analyzer.update()

            bass_level = analyzer.bass
            treble_level = analyzer.treble

            # ------------------------------------------------
            # UPDATE PARTICLES
            # ------------------------------------------------

            for particle in particles:
                particle.width = width
                particle.height = height

                particle.update(
                    bass_level,
                    dt,
                )

            # ------------------------------------------------
            # DRAW BACKGROUND
            # ------------------------------------------------

            screen.fill(
                BACKGROUND_COLOR
            )

            animation_time = time.monotonic()

            # ------------------------------------------------
            # DRAW BASS RINGS
            # ------------------------------------------------

            draw_bass_rings(
                screen,
                center_x,
                center_y,
                base_radius=BASS_RING_BASE_RADIUS,
                bass_level=bass_level,
                animation_time=animation_time,
            )

            # ------------------------------------------------
            # DRAW TREBLE WAVE
            # ------------------------------------------------

            draw_treble_wave(
                screen,
                center_y,
                width,
                treble_level,
                animation_time,
            )

            # ------------------------------------------------
            # DRAW PARTICLES
            # ------------------------------------------------

            for particle in particles:
                particle.draw(
                    screen,
                    center_x,
                    center_y,
                )

            # ------------------------------------------------
            # DRAW CENTER DOT
            # ------------------------------------------------

            center_radius = int(
                5 + bass_level * 12
            )

            pygame.draw.circle(
                screen,
                GREEN,
                (center_x, center_y),
                center_radius,
            )

            # ------------------------------------------------
            # DRAW HUD
            # ------------------------------------------------

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
            # DIAGNOSTIC LOGGING
            # ------------------------------------------------

            now = time.monotonic()

            if now - last_log_time >= LOG_INTERVAL:
                logger.info(
                    "Audio diagnostics | "
                    "callbacks=%s | callback_frames=%s | "
                    "bytes=%s | peak=%.6f | RMS=%.6f | "
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
        # Stop capture before shutting down Pygame.
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
