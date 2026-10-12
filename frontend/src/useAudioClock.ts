import { useEffect, useState } from "react";

/** Current playback time of an <audio> element, updated every animation frame
 * while playing (the element's own `timeupdate` event fires only ~4x/second,
 * far too coarse for a scrolling DJ waveform). `timeupdate` is still wired as a
 * fallback: browsers stop animation frames when the window isn't being drawn
 * (seen in testing: 0 frames/s), and the display should degrade, not freeze. */
export function useAudioClock(audio: HTMLAudioElement | null): number {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!audio) return;
    let raf = 0;
    const tick = () => {
      setT(audio.currentTime);
      if (!audio.paused) raf = requestAnimationFrame(tick);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const once = () => setT(audio.currentTime);
    audio.addEventListener("play", start);
    audio.addEventListener("seeked", once);
    audio.addEventListener("pause", once);
    audio.addEventListener("loadedmetadata", once);
    audio.addEventListener("timeupdate", once);
    return () => {
      cancelAnimationFrame(raf);
      audio.removeEventListener("play", start);
      audio.removeEventListener("seeked", once);
      audio.removeEventListener("pause", once);
      audio.removeEventListener("loadedmetadata", once);
      audio.removeEventListener("timeupdate", once);
    };
  }, [audio]);
  return t;
}
