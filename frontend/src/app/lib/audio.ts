/**
 * audio.ts — Synthetic UI Audio Engine for Project Sambhav.
 * Uses Web Audio API to generate procedural sounds without external assets.
 */

let audioCtx: AudioContext | null = null;

function getCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  }
  return audioCtx;
}

/**
 * Procedural UI sounds to make the application feel alive.
 */
export const sounds = {
  /** 
   * The "Tactile" Click 
   * A professional, short percussive snap combining high-frequency noise and a low sine-thud.
   */
  click: () => {
    const ctx = getCtx();
    const now = ctx.currentTime;

    // Part 1: High-end snap (filtered noise)
    const bufferSize = ctx.sampleRate * 0.02;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;

    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    const noiseFilter = ctx.createBiquadFilter();
    noiseFilter.type = 'highpass';
    noiseFilter.frequency.setValueAtTime(3000, now);
    
    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.04, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.02);

    noise.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(ctx.destination);

    // Part 2: Low-end "Thump" (sine)
    const osc = ctx.createOscillator();
    const oscGain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(300, now);
    osc.frequency.exponentialRampToValueAtTime(100, now + 0.04);
    
    oscGain.gain.setValueAtTime(0.06, now);
    oscGain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);

    osc.connect(oscGain);
    oscGain.connect(ctx.destination);

    noise.start(now);
    osc.start(now);
    osc.stop(now + 0.05);
  },

  /** 
   * The "Crystalline" Success 
   * A lush C Major 9 chord with a shimmering attack and soft exponential decay.
   */
  success: () => {
    const ctx = getCtx();
    const now = ctx.currentTime;
    
    // Cmaj9 frequencies: C5, E5, G5, B5, D6
    [523.25, 659.25, 783.99, 987.77, 1174.66].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now + i * 0.02); // slight stagger
      
      gain.gain.setValueAtTime(0, now + i * 0.02);
      gain.gain.linearRampToValueAtTime(0.03, now + i * 0.02 + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 1.2);
      
      osc.connect(gain);
      gain.connect(ctx.destination);
      
      osc.start(now + i * 0.02);
      osc.stop(now + 1.3);
    });
  },

  /** 
   * The "Subtle" Notify
   * A gentle resonant pop, designed to be helpful but non-intrusive.
   */
  notify: () => {
    const ctx = getCtx();
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(440, now);
    osc.frequency.exponentialRampToValueAtTime(880, now + 0.1);

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(2000, now);
    filter.frequency.exponentialRampToValueAtTime(500, now + 0.1);
    filter.Q.setValueAtTime(10, now);

    gain.gain.setValueAtTime(0.02, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.15);
  },

  /** 
   * The "Haptic" Error
   * A low-frequency damped thud that feels more like a mechanical vibration than a beep.
   */
  error: () => {
    const ctx = getCtx();
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(80, now);
    osc.frequency.exponentialRampToValueAtTime(40, now + 0.2);

    gain.gain.setValueAtTime(0.1, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.2);
  }
};
