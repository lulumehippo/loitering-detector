import jetson_inference
import jetson_utils
import time
import math
import argparse

# =============================================================================
#  JARVIS LOITERING DETECTOR  --  Iron Man / Stark Industries Theme
#  Jetson Nano | dustynv/jetson-inference:r35.4.1
# =============================================================================

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PERSON_CLASS_ID   = 1
CONFIDENCE_THRESH = 0.50
LOITER_THRESHOLD  = 7.0
WARNING_RATIO     = 0.6
TRACK_MIN_FRAMES  = 3
TRACK_DROP_FRAMES = 15
TRACK_OVERLAP     = 0.5
SCAN_SPEED        = 3
PANEL_WIDTH       = 280   # right panel pixel width

# ─────────────────────────────────────────────
# IRON MAN / JARVIS COLOR PALETTE  (R, G, B, A)
# ─────────────────────────────────────────────

C_CYAN      = (  0, 230, 255, 255)   # primary JARVIS cyan
C_CYAN_MID  = (  0, 190, 215, 180)   # secondary cyan
C_GOLD      = (255, 200,  50, 255)   # Stark gold
C_AMBER     = (255, 145,   0, 240)   # warning
C_RED       = (255,  40,  40, 255)   # threat / anomaly
C_WHITE     = (255, 255, 255, 220)
C_PANEL_BG  = (  0,   6,  18, 225)   # near-black navy for panel
C_ROW_ALT   = (  0,  20,  42,  90)   # alternating table row tint
C_DARK_BG   = (  0,   0,   0, 175)   # label background
C_DIVIDER   = (  0, 200, 230,  85)   # horizontal rule


# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="JARVIS Loitering Detector")
    p.add_argument("--input",     default="/dev/video0")
    p.add_argument("--output",    default="display://0")
    p.add_argument("--threshold", type=float, default=LOITER_THRESHOLD)
    return p.parse_args()


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────

def iron_bracket(img, det, color, t=2):
    """
    Iron Man targeting reticle: corner brackets + small center crosshair.
    """
    x1, y1 = int(det.Left),  int(det.Top)
    x2, y2 = int(det.Right), int(det.Bottom)
    arm = min(int(min(x2-x1, y2-y1) * 0.22), 28)

    jetson_utils.cudaDrawLine(img, (x1,      y1), (x1+arm,  y1), color, t)
    jetson_utils.cudaDrawLine(img, (x1,      y1), (x1,  y1+arm), color, t)
    jetson_utils.cudaDrawLine(img, (x2-arm,  y1), (x2,      y1), color, t)
    jetson_utils.cudaDrawLine(img, (x2,      y1), (x2,  y1+arm), color, t)
    jetson_utils.cudaDrawLine(img, (x1,  y2-arm), (x1,      y2), color, t)
    jetson_utils.cudaDrawLine(img, (x1,      y2), (x1+arm,  y2), color, t)
    jetson_utils.cudaDrawLine(img, (x2-arm,  y2), (x2,      y2), color, t)
    jetson_utils.cudaDrawLine(img, (x2,  y2-arm), (x2,      y2), color, t)

    # center crosshair
    cx, cy = (x1+x2)//2, (y1+y2)//2
    jetson_utils.cudaDrawLine(img, (cx-5, cy), (cx+5, cy), color, 1)
    jetson_utils.cudaDrawLine(img, (cx, cy-5), (cx, cy+5), color, 1)


def progress_bar(img, det, elapsed, threshold):
    x1    = int(det.Left)
    y2    = int(det.Bottom)
    w     = max(int(det.Width), 4)
    ratio = min(elapsed / threshold, 1.0)
    filled = int(w * ratio)

    jetson_utils.cudaDrawRect(img, (x1, y2+3, x1+w, y2+9), (5, 15, 28, 190))
    if filled > 0:
        col = C_RED if ratio >= 1.0 else C_AMBER if ratio >= WARNING_RATIO else C_CYAN
        jetson_utils.cudaDrawRect(img, (x1, y2+3, x1+filled, y2+9), col)


def draw_arc_reactor(img, cx, cy):
    """
    Concentric filled disks drawn largest-first so smaller inner disks
    paint on top, creating a layered glow that looks like an arc reactor.
    cudaDrawCircle draws filled disks.
    """
    try:
        jetson_utils.cudaDrawCircle(img, (cx, cy), 46, ( 0, 140, 170,  45))
        jetson_utils.cudaDrawCircle(img, (cx, cy), 38, ( 0, 170, 200,  70))
        jetson_utils.cudaDrawCircle(img, (cx, cy), 28, ( 0, 200, 230, 110))
        jetson_utils.cudaDrawCircle(img, (cx, cy), 18, ( 0, 218, 248, 160))
        jetson_utils.cudaDrawCircle(img, (cx, cy), 10, ( 0, 230, 255, 210))
        # 8 radial spokes between r=13 and r=25
        for i in range(8):
            angle = i * math.pi / 4.0
            ax1 = int(cx + 13 * math.cos(angle))
            ay1 = int(cy + 13 * math.sin(angle))
            ax2 = int(cx + 25 * math.cos(angle))
            ay2 = int(cy + 25 * math.sin(angle))
            jetson_utils.cudaDrawLine(img, (ax1, ay1), (ax2, ay2),
                                       (0, 230, 255, 130), 1)
    except Exception:
        # cudaDrawCircle unavailable in this build -- skip gracefully
        pass


def hline(img, y, x1, x2, color=C_DIVIDER, thickness=1):
    jetson_utils.cudaDrawLine(img, (x1, y), (x2, y), color, thickness)


def vline(img, x, y1, y2, color=C_CYAN, thickness=1):
    jetson_utils.cudaDrawLine(img, (x, y1), (x, y2), color, thickness)


# ─────────────────────────────────────────────
# HUD ELEMENTS
# ─────────────────────────────────────────────

def draw_top_hud(img, font, fps):
    W  = img.width
    H  = img.height
    px = W - PANEL_WIDTH

    jetson_utils.cudaDrawRect(img, (0, 0, W, 38), (0, 4, 12, 210))
    hline(img, 38, 0, W, C_CYAN, 1)

    font.OverlayText(img, W, H,
        "[ SECURITY SYSTEM ]",
        10, 10, C_CYAN, (0, 0, 0, 0))
    font.OverlayText(img, W, H,
        f"NET {fps:.0f} FPS",
        px - 110, 10, C_CYAN_MID, (0, 0, 0, 0))


def draw_right_panel(img, font, track_data, num_threats, total_anomalies):
    """
    Right-side JARVIS panel:
      Arc reactor header | stats | live tracking table | anomaly footer
    track_data: list of (tid, elapsed, status_str, color) sorted by elapsed desc
    """
    W  = img.width
    H  = img.height
    px = W - PANEL_WIDTH

    # ── Panel background + borders ────────────────────────────────────────
    jetson_utils.cudaDrawRect(img, (px, 0, W, H), C_PANEL_BG)
    vline(img, px,   0, H, C_CYAN, 2)
    vline(img, px+2, 0, H, (0, 100, 130, 60), 1)

    # ── Arc reactor ───────────────────────────────────────────────────────
    arc_cx = px + PANEL_WIDTH // 2
    arc_cy = 90
    draw_arc_reactor(img, arc_cx, arc_cy)

    # ── Header text ───────────────────────────────────────────────────────
    font.OverlayText(img, W, H, "J.A.R.V.I.S.",
                     px + 112, 22, C_CYAN, (0, 0, 0, 0))
    font.OverlayText(img, W, H, "THREAT DETECTION v2.0",
                     px + 65,  46, C_CYAN_MID, (0, 0, 0, 0))

    # ── Divider 1 ─────────────────────────────────────────────────────────
    y1 = 148
    hline(img, y1, px+10, W-10)

    # ── Stats row ─────────────────────────────────────────────────────────
    font.OverlayText(img, W, H, f"SUBJECTS: {len(track_data)}",
                     px + 14, y1 + 12, C_CYAN, (0, 0, 0, 0))
    font.OverlayText(img, W, H, f"THREATS: {num_threats}",
                     px + 215, y1 + 12,
                     C_RED if num_threats > 0 else C_CYAN,
                     (0, 0, 0, 0))

    # ── Divider 2 ─────────────────────────────────────────────────────────
    y2 = y1 + 40
    hline(img, y2, px+10, W-10)

    # ── Table column headers ───────────────────────────────────────────────
    yth = y2 + 12
    font.OverlayText(img, W, H, "ID",      px + 14,  yth, C_GOLD, (0,0,0,0))
    font.OverlayText(img, W, H, "TIME(s)", px + 60,  yth, C_GOLD, (0,0,0,0))
    font.OverlayText(img, W, H, "STATUS",  px + 140, yth, C_GOLD, (0,0,0,0))

    y3 = yth + 26
    hline(img, y3, px+10, W-10)

    # ── Table rows ────────────────────────────────────────────────────────
    row_h    = 22
    y_row    = y3 + 8
    max_rows = (H - y_row - 60) // row_h

    for i, (tid, elapsed, status, color) in enumerate(track_data[:max_rows]):
        if i % 2 == 0:
            jetson_utils.cudaDrawRect(img,
                (px+10, y_row-4, W-10, y_row + row_h - 6),
                C_ROW_ALT)

        font.OverlayText(img, W, H, str(tid),
                         px + 14,  y_row, color, (0,0,0,0))
        font.OverlayText(img, W, H, f"{elapsed:>6.1f}",
                         px + 60,  y_row, color, (0,0,0,0))
        font.OverlayText(img, W, H, status,
                         px + 140, y_row, color, (0,0,0,0))
        y_row += row_h

    # ── Footer ────────────────────────────────────────────────────────────
    y_foot = H - 48
    hline(img, y_foot, px+10, W-10)
    font.OverlayText(img, W, H,
        f"TOTAL ANOMALIES: {total_anomalies}",
        px + 55, y_foot + 14, C_GOLD, (0,0,0,0))


def draw_alert_banner(img, font, count, flash):
    """Blinking red banner in the camera area (not over the panel)."""
    if count == 0 or not flash:
        return
    H  = img.height
    W  = img.width
    px = W - PANEL_WIDTH
    jetson_utils.cudaDrawRect(img, (0, H-44, px, H), (140, 0, 0, 210))
    font.OverlayText(img, W, H,
        f"!! THREAT CONFIRMED -- {count} TARGET(S) LOITERING !!",
        10, H-32, C_WHITE, (0, 0, 0, 0))


def draw_scan_line(img, y):
    """Subtle animated horizontal scan line in the camera area only."""
    px = img.width - PANEL_WIDTH
    jetson_utils.cudaDrawLine(img, (0, y), (px, y), (0, 200, 230, 55), 1)


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class JARVISDetector:

    def __init__(self, args):
        self.loiter_threshold  = args.threshold
        self.warning_threshold = self.loiter_threshold * WARNING_RATIO

        self.track_time = {}
        self.loiter_ids = set()
        self.total      = 0
        self.scan_y     = 40

        print("[JARVIS] Initializing neural network...")
        self.net = jetson_inference.detectNet(
            "ssd-mobilenet-v2", threshold=CONFIDENCE_THRESH)
        self.net.SetTrackingEnabled(True)
        self.net.SetTrackingParams(
            minFrames=TRACK_MIN_FRAMES,
            dropFrames=TRACK_DROP_FRAMES,
            overlapThreshold=TRACK_OVERLAP
        )

        self.cam = jetson_utils.videoSource(args.input,
            argv=['--input-width=960', '--input-height=540'])
        self.disp = jetson_utils.videoOutput(args.output)
        self.font = jetson_utils.cudaFont(size=18.0)
        print(f"[JARVIS] Online. Threshold: {self.loiter_threshold:.1f}s")

    def run(self):
        while self.disp.IsStreaming():
            img = self.cam.Capture()
            if img is None:
                continue

            now        = time.time()
            detections = self.net.Detect(img, overlay="none")

            active     = set()
            threats    = 0
            track_data = []   # (tid, elapsed, status_str, color)

            for det in detections:
                if det.ClassID != PERSON_CLASS_ID or det.TrackStatus < 0:
                    continue

                tid = det.TrackID
                active.add(tid)

                if tid not in self.track_time:
                    self.track_time[tid] = now

                elapsed = now - self.track_time[tid]

                if elapsed >= self.loiter_threshold:
                    color  = C_RED
                    status = "THREAT"
                    threats += 1
                    if tid not in self.loiter_ids:
                        self.loiter_ids.add(tid)
                        self.total += 1
                        print(f"[JARVIS] THREAT: ID {tid} -- {elapsed:.1f}s")

                elif elapsed >= self.warning_threshold:
                    color  = C_AMBER
                    status = "WARNING"

                else:
                    color  = C_CYAN
                    status = "OK"

                track_data.append((tid, elapsed, status, color))

                # Draw overlay on camera frame
                iron_bracket(img, det, color)
                progress_bar(img, det, elapsed, self.loiter_threshold)
                self.font.OverlayText(
                    img, img.width, img.height,
                    f"ID:{tid}  {elapsed:.1f}s",
                    int(det.Left),
                    max(int(det.Top) - 22, 40),
                    color, C_DARK_BG
                )

            # Cleanup dropped tracks
            for tid in list(self.track_time.keys()):
                if tid not in active:
                    del self.track_time[tid]
                    self.loiter_ids.discard(tid)

            # Sort: highest elapsed first (threats at top of table)
            track_data.sort(key=lambda x: x[1], reverse=True)

            # HUD
            self.scan_y = (self.scan_y + SCAN_SPEED) % img.height
            draw_scan_line(img, self.scan_y)
            draw_top_hud(img, self.font, self.net.GetNetworkFPS())
            draw_right_panel(img, self.font, track_data, threats, self.total)
            flash = (now % 1.0) < 0.6
            draw_alert_banner(img, self.font, threats, flash)

            self.disp.Render(img)
            self.disp.SetStatus(
                f"JARVIS | Subjects:{len(track_data)} "
                f"Threats:{threats} Anomalies:{self.total}"
            )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    JARVISDetector(args).run()


if __name__ == "__main__":
    main()
