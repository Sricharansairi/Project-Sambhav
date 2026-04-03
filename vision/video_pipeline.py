import os, cv2, logging, tempfile, time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── Frame extractor ───────────────────────────────────────────
def extract_frames(video_path: str, fps: int = 1) -> list:
    """Extract 1 frame per second from video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap    = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval  = max(1, int(video_fps / fps))
    frames, frame_idx, saved = [], 0, []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % interval == 0:
            # Save frame as temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, frame)
            frames.append({
                "path":      tmp.name,
                "frame_idx": frame_idx,
                "timestamp": round(frame_idx / video_fps, 2),
            })
        frame_idx += 1
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames

# ── Analyze single frame ──────────────────────────────────────
def analyze_frame(frame_info: dict, domain: str) -> dict:
    """Analyze a single frame using image pipeline."""
    try:
        from vision.image_pipeline import analyze_image
        result = analyze_image(frame_info["path"], domain)
        result["timestamp"] = frame_info["timestamp"]
        result["frame_idx"] = frame_info["frame_idx"]
        return result
    except Exception as e:
        logger.warning(f"Frame analysis failed at {frame_info['timestamp']}s: {e}")
        return {
            "timestamp":        frame_info["timestamp"],
            "frame_idx":        frame_info["frame_idx"],
            "dominant_emotion": "neutral",
            "stress_level":     0.5,
            "engagement_level": 0.5,
            "energy_level":     0.5,
            "error":            str(e),
        }

# ── Emotional timeline builder ────────────────────────────────
def build_timeline(frame_results: list) -> list:
    """
    Build color-coded emotional timeline.
    Green=calm, Amber=tension, Red=conflict,
    Blue=analytical, Purple=dominant
    """
    timeline = []
    for fr in frame_results:
        emotion = fr.get("dominant_emotion", "neutral")
        stress  = fr.get("stress_level", 0.5)

        # Color coding
        if stress < 0.3 and emotion in ["happy","neutral"]:
            color = "green"
            label = "calm"
        elif stress < 0.5:
            color = "blue"
            label = "analytical"
        elif stress < 0.7 and emotion in ["angry","fear"]:
            color = "amber"
            label = "tension"
        elif stress >= 0.7 and emotion in ["angry","disgust"]:
            color = "red"
            label = "conflict"
        elif emotion in ["surprise"]:
            color = "purple"
            label = "dominant"
        else:
            color = "blue"
            label = "neutral"

        timeline.append({
            "timestamp":  fr.get("timestamp", 0),
            "emotion":    emotion,
            "stress":     stress,
            "engagement": fr.get("engagement_level", 0.5),
            "color":      color,
            "label":      label,
        })
    return timeline

# ── Key moment detector ───────────────────────────────────────
def detect_key_moments(timeline: list) -> list:
    """Detect significant moments in the video."""
    if not timeline: return []
    moments = []
    prev_stress = timeline[0]["stress"]

    for i, frame in enumerate(timeline):
        stress = frame["stress"]
        # Tension spike
        if stress - prev_stress > 0.25:
            moments.append({
                "timestamp": frame["timestamp"],
                "type":      "tension_spike",
                "label":     f"Tension spike at {frame['timestamp']}s",
                "stress":    stress,
            })
        # Conflict threshold crossed
        if stress >= 0.8 and prev_stress < 0.8:
            moments.append({
                "timestamp": frame["timestamp"],
                "type":      "conflict_start",
                "label":     f"Conflict begins at {frame['timestamp']}s",
                "stress":    stress,
            })
        # Calm recovery
        if stress < 0.3 and prev_stress >= 0.5:
            moments.append({
                "timestamp": frame["timestamp"],
                "type":      "calm_recovery",
                "label":     f"Calm recovery at {frame['timestamp']}s",
                "stress":    stress,
            })
        prev_stress = stress

    return moments

# ── Aggregate frame results ───────────────────────────────────
def aggregate_results(frame_results: list) -> dict:
    """Aggregate all frame results into single prediction input."""
    if not frame_results:
        return {}
    stresses    = [f.get("stress_level", 0.5)     for f in frame_results]
    engagements = [f.get("engagement_level", 0.5) for f in frame_results]
    energies    = [f.get("energy_level", 0.5)      for f in frame_results]
    emotions    = [f.get("dominant_emotion","neutral") for f in frame_results]

    # Most common emotion
    from collections import Counter
    dominant_emotion = Counter(emotions).most_common(1)[0][0]

    # Reliability — based on frame coverage
    total_duration = frame_results[-1].get("timestamp", 0)
    frame_count    = len(frame_results)
    reliability    = min(1.0, frame_count / max(total_duration, 1))

    return {
        "avg_stress":        round(float(np.mean(stresses)), 3),
        "max_stress":        round(float(np.max(stresses)), 3),
        "avg_engagement":    round(float(np.mean(engagements)), 3),
        "avg_energy":        round(float(np.mean(energies)), 3),
        "dominant_emotion":  dominant_emotion,
        "emotion_variance":  round(float(np.std(stresses)), 3),
        "frames_analyzed":   frame_count,
        "total_duration":    total_duration,
        "frame_reliability": round(reliability, 3),
        "inferred_parameters": {
            "stress_level":     round(float(np.mean(stresses)), 3),
            "engagement_level": round(float(np.mean(engagements)), 3),
            "energy_level":     round(float(np.mean(energies)), 3),
            "dominant_emotion": dominant_emotion,
            "peak_stress":      round(float(np.max(stresses)), 3),
            "stress_variance":  round(float(np.std(stresses)), 3),
        }
    }

# ── Speed modes ───────────────────────────────────────────────
SPEED_MODES = {
    "fast":     {"fps": 0.5, "label": "Fast (every 2nd frame, 15-20s)"},
    "standard": {"fps": 1,   "label": "Standard (every frame, 25-40s)"},
    "deep":     {"fps": 2,   "label": "Deep (all frames, 60-90s)"},
}

# ── MAIN ENTRY POINT ──────────────────────────────────────────
def analyze_video(
    video_path: str,
    domain:     str  = "general",
    mode:       str  = "standard",
    callback          = None,
) -> dict:
    """
    Full video analysis pipeline.
    callback(progress, partial_result) called after each frame — 
    enables live Streamlit updates.
    """
    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}

    fps = SPEED_MODES.get(mode, SPEED_MODES["standard"])["fps"]
    logger.info(f"Analyzing video: {video_path}, mode={mode}, fps={fps}")

    # Extract frames
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return {"error": "No frames extracted from video"}

    # Analyze each frame
    frame_results = []
    for i, frame in enumerate(frames):
        result = analyze_frame(frame, domain)
        frame_results.append(result)

        # Live callback for Streamlit progress
        if callback:
            progress = (i + 1) / len(frames)
            partial  = aggregate_results(frame_results)
            callback(progress, partial)

        # Clean up temp file
        try: os.unlink(frame["path"])
        except: pass

    # Build outputs
    timeline    = build_timeline(frame_results)
    key_moments = detect_key_moments(timeline)
    aggregated  = aggregate_results(frame_results)

    return {
        "video_path":    video_path,
        "domain":        domain,
        "mode":          mode,
        "frames":        len(frame_results),
        "timeline":      timeline,
        "key_moments":   key_moments,
        "aggregated":    aggregated,
        "inferred_parameters": aggregated.get("inferred_parameters", {}),
        "frame_results": frame_results,
    }

if __name__ == "__main__":
    import sys
    path   = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    domain = sys.argv[2] if len(sys.argv) > 2 else "general"
    mode   = sys.argv[3] if len(sys.argv) > 3 else "fast"
    result = analyze_video(path, domain, mode)
    print(f"\n🎥 VIDEO ANALYSIS RESULT:")
    print(f"  Frames analyzed : {result.get('frames')}")
    print(f"  Avg stress      : {result['aggregated'].get('avg_stress')}")
    print(f"  Avg engagement  : {result['aggregated'].get('avg_engagement')}")
    print(f"  Dominant emotion: {result['aggregated'].get('dominant_emotion')}")
    print(f"  Key moments     : {len(result.get('key_moments',[]))}")
