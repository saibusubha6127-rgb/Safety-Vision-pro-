import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

// ==========================================
// 1. GLOBAL TYPES & CONFIGURATION
// ==========================================
declare global {
  interface Window {
    FaceMesh: any;
    cocoSsd: any;
    tf: any;
  }
}

const CONFIG = {
  EAR_THRESHOLD: 0.20,
  MAR_THRESHOLD: 0.50,
  BLINK_FRAMES: 5,
  DROWSY_FRAMES: 15,
  PERCLOS_WINDOW: 60000,
  SAFE_DISTANCE: 8.0,
  KNOWN_WIDTH: 2.5,
  DEFAULT_FOCAL: 600,
  HEAVY_VEHICLES: ['bus', 'truck'],
  TRAFFIC_INTERVAL_MS: 150, // Slightly increased for stability
  LOG_LIMIT: 1000,
};

// ==========================================
// 2. MODULAR ENGINES & UTILS
// ==========================================

interface LogEntry {
  timestamp: string;
  type: 'INFO' | 'WARNING' | 'CRITICAL';
  message: string;
  data?: string;
}

class SystemLogger {
  private logs: LogEntry[] = [];
  log(type: 'INFO' | 'WARNING' | 'CRITICAL', message: string, data?: string) {
    const entry: LogEntry = { timestamp: new Date().toISOString(), type, message, data: data || '' };
    this.logs.unshift(entry);
    if (this.logs.length > CONFIG.LOG_LIMIT) this.logs.pop();
    console.log(`[${type}] ${message}`, data || '');
  }
  downloadCSV() {
    const headers = ['Timestamp', 'Type', 'Message', 'Data'];
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...this.logs.map(e => `${e.timestamp},${e.type},"${e.message}","${e.data}"`)].join('\n');
    const link = document.createElement("a");
    link.setAttribute("href", encodeURI(csvContent));
    link.setAttribute("download", `safety_vision_logs_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}

const logger = new SystemLogger();

class AlertSystem {
  private audioCtx: AudioContext | null = null;
  private lastSpeechTime = 0;
  init() {
    if (!this.audioCtx) this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  }
  beep(frequency = 1000, duration = 200, type: OscillatorType = 'square') {
    if (!this.audioCtx) return;
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    osc.connect(gain); gain.connect(this.audioCtx.destination);
    osc.type = type; osc.frequency.value = frequency; gain.gain.value = 0.05;
    osc.start(); setTimeout(() => osc.stop(), duration);
  }
  speak(text: string) {
    const now = Date.now();
    if (now - this.lastSpeechTime < 5000) return;
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.1;
      window.speechSynthesis.speak(utterance);
      this.lastSpeechTime = now;
    }
  }
}

const alertSystem = new AlertSystem();
const euclidean = (p1: any, p2: any) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

class DrowsinessEngine {
  private eyeHistory: {timestamp: number, closed: boolean}[] = [];
  calculateEAR(landmarks: any[]) {
    const ear = (indices: number[]) => {
      const v1 = euclidean(landmarks[indices[1]], landmarks[indices[5]]);
      const v2 = euclidean(landmarks[indices[2]], landmarks[indices[4]]);
      const h = euclidean(landmarks[indices[0]], landmarks[indices[3]]);
      return (v1 + v2) / (2.0 * h);
    };
    return (ear([33, 160, 158, 133, 153, 144]) + ear([362, 385, 387, 263, 373, 380])) / 2.0;
  }
  calculateMAR(landmarks: any[]) {
    return euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[78], landmarks[308]);
  }
  calculatePERCLOS() {
    const now = Date.now();
    this.eyeHistory = this.eyeHistory.filter(e => now - e.timestamp < CONFIG.PERCLOS_WINDOW);
    return this.eyeHistory.length === 0 ? 0 : this.eyeHistory.filter(e => e.closed).length / this.eyeHistory.length;
  }
  updateHistory(isClosed: boolean) { this.eyeHistory.push({ timestamp: Date.now(), closed: isClosed }); }
}

class TrafficEngine {
  private history: number[] = [];
  estimateDistance(pixelWidth: number, focalLength: number) {
    return pixelWidth <= 0 ? null : (CONFIG.KNOWN_WIDTH * focalLength) / pixelWidth;
  }
  smoothDistance(newDist: number) {
    this.history.push(newDist);
    if (this.history.length > 5) this.history.shift();
    return this.history.reduce((a, b) => a + b, 0) / this.history.length;
  }
}

const drowsyEngine = new DrowsinessEngine();
const trafficEngine = new TrafficEngine();

// ==========================================
// 3. MAIN COMPONENT
// ==========================================
const App = () => {
  const [driverDeviceId, setDriverDeviceId] = useState<string>('');
  const [rearDeviceId, setRearDeviceId] = useState<string>('');
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [systemStarted, setSystemStarted] = useState(false);
  
  const [fps, setFps] = useState(0);
  const [ear, setEar] = useState(0);
  const [mar, setMar] = useState(0);
  const [perclos, setPerclos] = useState(0);
  const [focalLength, setFocalLength] = useState(CONFIG.DEFAULT_FOCAL);
  
  const [driverStatus, setDriverStatus] = useState<'AWAKE' | 'DROWSY' | 'YAWNING'>('AWAKE');
  const [rearStatus, setRearStatus] = useState<'SAFE' | 'WARNING' | 'CRITICAL'>('SAFE');
  const [detectedObj, setDetectedObj] = useState<{label: string, dist: number} | null>(null);

  const [isDetecting, setIsDetecting] = useState(true);
  const [activeView, setActiveView] = useState<'BOTH' | 'DRIVER' | 'REAR'>('BOTH');

  const driverVideo = useRef<HTMLVideoElement>(null);
  const rearVideo = useRef<HTMLVideoElement>(null);
  const driverCanvas = useRef<HTMLCanvasElement>(null);
  const rearCanvas = useRef<HTMLCanvasElement>(null);
  
  const faceMeshRef = useRef<any>(null);
  const cocoRef = useRef<any>(null);
  const earCounter = useRef(0);
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  const lastTrafficProcess = useRef(0);
  
  const isDetectingRef = useRef(true);
  const activeViewRef = useRef<'BOTH' | 'DRIVER' | 'REAR'>('BOTH');

  useEffect(() => { isDetectingRef.current = isDetecting; }, [isDetecting]);
  useEffect(() => { activeViewRef.current = activeView; }, [activeView]);

  // 1. Initialization
  useEffect(() => {
    const init = async () => {
      try {
        if (window.tf) await window.tf.setBackend('webgl');
        
        const faceMesh = new window.FaceMesh({ locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
        faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        faceMesh.onResults(onFaceResults);
        faceMeshRef.current = faceMesh;

        cocoRef.current = await window.cocoSsd.load();

        const devs = await navigator.mediaDevices.enumerateDevices();
        const videos = devs.filter(d => d.kind === 'videoinput');
        setDevices(videos);
        if (videos.length > 0) {
            setDriverDeviceId(videos[0].deviceId);
            setRearDeviceId(videos.length > 1 ? videos[1].deviceId : videos[0].deviceId);
        }

        setModelsLoaded(true);
        logger.log('INFO', 'Models initialized');
      } catch (e) {
        logger.log('CRITICAL', 'Initialization Error');
      }
    };
    init();
  }, []);

  // 2. Camera Start Logic
  const startCameras = async () => {
    alertSystem.init();
    setSystemStarted(true);
    logger.log('INFO', 'Attempting to start cameras...');
  };

  useEffect(() => {
    if (!systemStarted || !modelsLoaded) return;

    const setupStream = async (videoRef: React.RefObject<HTMLVideoElement>, deviceId: string) => {
      if (!videoRef.current) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId: { exact: deviceId } } : true
        });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        logger.log('INFO', `Stream started for device ${deviceId.slice(0, 8)}`);
      } catch (err) {
        logger.log('WARNING', `Failed specific camera, trying default...`);
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        } catch (e) {
          logger.log('CRITICAL', 'Camera access denied');
        }
      }
    };

    // Small delay to ensure refs are attached after the "systemStarted" render
    setTimeout(() => {
      setupStream(driverVideo, driverDeviceId);
      setupStream(rearVideo, rearDeviceId);
    }, 100);

  }, [systemStarted, modelsLoaded, driverDeviceId, rearDeviceId]);

  // 3. Main Logic Loop
  useEffect(() => {
    if (!systemStarted || !modelsLoaded) return;
    let animId: number;

    const loop = async () => {
      const now = performance.now();
      frameCount.current++;
      if (now - lastTime.current >= 1000) {
        setFps(frameCount.current);
        frameCount.current = 0;
        lastTime.current = now;
      }

      if (isDetectingRef.current) {
        if ((activeViewRef.current === 'BOTH' || activeViewRef.current === 'DRIVER') && driverVideo.current && driverVideo.current.readyState === 4) {
          await faceMeshRef.current.send({ image: driverVideo.current });
        }

        if ((activeViewRef.current === 'BOTH' || activeViewRef.current === 'REAR') && now - lastTrafficProcess.current > CONFIG.TRAFFIC_INTERVAL_MS) {
          await processTraffic();
          lastTrafficProcess.current = now;
        }
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [systemStarted, modelsLoaded, focalLength]);

  const onFaceResults = (results: any) => {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
    const landmarks = results.multiFaceLandmarks[0];
    
    const currentEar = drowsyEngine.calculateEAR(landmarks);
    const currentMar = drowsyEngine.calculateMAR(landmarks);
    const isClosed = currentEar < CONFIG.EAR_THRESHOLD;
    
    drowsyEngine.updateHistory(isClosed);
    const currentPerclos = drowsyEngine.calculatePERCLOS();

    setEar(currentEar); setMar(currentMar); setPerclos(currentPerclos);

    let status: 'AWAKE' | 'DROWSY' | 'YAWNING' = 'AWAKE';
    if (currentMar > CONFIG.MAR_THRESHOLD) status = 'YAWNING';

    if (isClosed) earCounter.current++;
    else if (earCounter.current > 0) earCounter.current--;

    if (earCounter.current > CONFIG.DROWSY_FRAMES || currentPerclos > 0.4) {
      status = 'DROWSY';
      alertSystem.beep(1000, 300, 'sawtooth');
      alertSystem.speak("Wake up!");
    }

    setDriverStatus(status);
    drawDriverOverlay(landmarks, status);
  };

  const drawDriverOverlay = (landmarks: any[], status: string) => {
    const canvas = driverCanvas.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = 640; canvas.height = 480;
    ctx.clearRect(0,0,640,480);
    const color = status === 'AWAKE' ? '#00ff00' : (status === 'YAWNING' ? '#ffff00' : '#ff0000');
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    [ [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380] ].forEach(eye => {
      ctx.beginPath();
      eye.forEach((idx, i) => {
        const x = landmarks[idx].x * 640; const y = landmarks[idx].y * 480;
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      });
      ctx.closePath(); ctx.stroke();
    });
  };

  const processTraffic = async () => {
    if (!rearVideo.current || !cocoRef.current || rearVideo.current.readyState !== 4) return;
    const predictions = await cocoRef.current.detect(rearVideo.current);
    const canvas = rearCanvas.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;

    canvas.width = rearVideo.current.videoWidth; canvas.height = rearVideo.current.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let closest: {label: string, dist: number} | null = null;
    let minDiff = Infinity;

    predictions.forEach((pred: any) => {
      if (CONFIG.HEAVY_VEHICLES.includes(pred.class)) {
        const [x, y, w, h] = pred.bbox;
        const rawDist = trafficEngine.estimateDistance(w, focalLength);
        if (rawDist !== null) {
          const smoothDist = trafficEngine.smoothDistance(rawDist);
          const isCritical = smoothDist < CONFIG.SAFE_DISTANCE;
          const color = isCritical ? '#ff0000' : '#00ff00';
          ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.strokeRect(x, y, w, h);
          ctx.fillStyle = color; ctx.font = 'bold 16px Arial';
          ctx.fillText(`${pred.class.toUpperCase()} ${smoothDist.toFixed(1)}m`, x, y - 10);
          if (smoothDist < minDiff) { minDiff = smoothDist; closest = { label: pred.class, dist: smoothDist }; }
        }
      }
    });

    setDetectedObj(closest);
    if (closest && closest.dist < CONFIG.SAFE_DISTANCE) {
      setRearStatus('CRITICAL');
      alertSystem.beep(1500, 200, 'square');
      alertSystem.speak("Vehicle alert");
    } else {
      setRearStatus(closest ? 'WARNING' : 'SAFE');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans flex flex-col">
      <header className="bg-gray-800 p-4 shadow-lg flex justify-between items-center z-10">
        <div>
          <h1 className="text-2xl font-bold text-blue-400">SafetyVision Pro</h1>
          <div className="text-xs text-gray-400 flex gap-4 mt-1">
            <span>FPS: <span className="text-green-400 font-mono">{fps}</span></span>
            <span>SYSTEM: <span className={systemStarted ? 'text-green-400' : 'text-yellow-400'}>{systemStarted ? 'ACTIVE' : 'READY'}</span></span>
          </div>
        </div>
        
        {systemStarted && (
          <div className="flex gap-4">
            <div className={`px-4 py-2 rounded-lg font-bold border-2 transition-all 
              ${driverStatus === 'AWAKE' ? 'bg-green-900/30 border-green-500 text-green-400' : 
                driverStatus === 'YAWNING' ? 'bg-yellow-900/30 border-yellow-500 text-yellow-400' : 
                'bg-red-900/50 border-red-500 text-red-500 animate-pulse'}`}>
               DRIVER: {driverStatus}
            </div>
            <div className={`px-4 py-2 rounded-lg font-bold border-2 transition-all
              ${rearStatus === 'SAFE' ? 'bg-gray-800 border-gray-600 text-gray-400' : 
                rearStatus === 'CRITICAL' ? 'bg-red-900/50 border-red-500 text-red-500 animate-pulse' : 
                'bg-blue-900/30 border-blue-500 text-blue-400'}`}>
               REAR: {rearStatus === 'CRITICAL' ? 'COLLISION' : (detectedObj ? `${detectedObj.dist.toFixed(1)}m` : 'CLEAR')}
            </div>
          </div>
        )}
      </header>

      {!systemStarted ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
           <div className="max-w-md bg-gray-800 p-8 rounded-2xl shadow-2xl border border-gray-700">
             <div className="mb-6 text-blue-400">
               <svg className="w-20 h-20 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
             </div>
             <h2 className="text-2xl font-bold mb-4">Start Safety Monitoring</h2>
             <p className="text-gray-400 mb-8">This application requires camera access to monitor driver state and rear traffic. All processing is done locally.</p>
             <button 
               onClick={startCameras} 
               disabled={!modelsLoaded}
               className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold py-4 rounded-xl shadow-lg transition-all transform hover:scale-105 active:scale-95"
             >
               {modelsLoaded ? 'START SYSTEM' : 'LOADING AI MODELS...'}
             </button>
           </div>
        </div>
      ) : (
        <main className="flex-1 p-4 flex flex-col gap-4 overflow-y-auto">
          <div className="flex flex-wrap justify-center gap-4 mb-2 bg-gray-800 p-3 rounded-xl border border-gray-700 shadow-md">
             <div className="flex bg-gray-900 rounded-lg p-1 border border-gray-700">
               <button onClick={() => setActiveView('BOTH')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${activeView === 'BOTH' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}>Dual View</button>
               <button onClick={() => setActiveView('DRIVER')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${activeView === 'DRIVER' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}>Driver Only</button>
               <button onClick={() => setActiveView('REAR')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${activeView === 'REAR' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}>Rear Only</button>
             </div>
             
             <button onClick={() => setIsDetecting(!isDetecting)} className={`px-6 py-1.5 rounded-lg text-sm font-bold transition-all shadow-md ${isDetecting ? 'bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30' : 'bg-green-500/20 text-green-400 border border-green-500/50 hover:bg-green-500/30'}`}>
               {isDetecting ? '⏸ PAUSE DETECTION' : '▶ RESUME DETECTION'}
             </button>
          </div>

          <div className={`grid grid-cols-1 ${activeView === 'BOTH' ? 'lg:grid-cols-2' : ''} gap-6`}>
          <div className={`flex flex-col gap-2 ${activeView === 'REAR' ? 'hidden' : ''}`}>
             <div className="flex justify-between items-center pb-2 border-b border-gray-700">
                <h2 className="text-lg font-semibold flex items-center gap-2">Driver Monitor</h2>
                <select className="bg-gray-800 text-xs p-1 rounded border border-gray-600" value={driverDeviceId} onChange={e => setDriverDeviceId(e.target.value)}>
                  {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || 'Camera '+d.deviceId.slice(0,4)}</option>)}
                </select>
             </div>
             <div className="relative bg-black rounded-lg overflow-hidden aspect-video border border-gray-700 shadow-xl">
               <video ref={driverVideo} autoPlay playsInline muted className="w-full h-full object-cover scale-x-[-1]" />
               <canvas ref={driverCanvas} className="absolute top-0 left-0 w-full h-full scale-x-[-1]" />
               <div className="absolute top-2 left-2 bg-black/60 p-2 rounded text-[10px] font-mono border border-white/10">
                 EAR: {ear.toFixed(2)} | PERCLOS: {(perclos*100).toFixed(0)}%
               </div>
               {!isDetecting && (
                 <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center">
                   <span className="text-xl font-bold text-yellow-400 tracking-widest border-2 border-yellow-400/50 px-4 py-2 rounded-lg bg-black/50">
                     PAUSED
                   </span>
                 </div>
               )}
             </div>
          </div>

          <div className={`flex flex-col gap-2 ${activeView === 'DRIVER' ? 'hidden' : ''}`}>
             <div className="flex justify-between items-center pb-2 border-b border-gray-700">
                <h2 className="text-lg font-semibold">Rear Monitor</h2>
                <select className="bg-gray-800 text-xs p-1 rounded border border-gray-600" value={rearDeviceId} onChange={e => setRearDeviceId(e.target.value)}>
                  {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || 'Camera '+d.deviceId.slice(0,4)}</option>)}
                </select>
             </div>
             <div className="relative bg-black rounded-lg overflow-hidden aspect-video border border-gray-700 shadow-xl">
               <video ref={rearVideo} autoPlay playsInline muted className="w-full h-full object-cover" />
               <canvas ref={rearCanvas} className="absolute top-0 left-0 w-full h-full" />
               {detectedObj && (
                 <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/80 px-4 py-2 rounded-full border border-red-500/50 flex gap-4 items-center">
                    <span className="font-bold">{detectedObj.label.toUpperCase()}</span>
                    <span className={`text-xl font-mono ${detectedObj.dist < CONFIG.SAFE_DISTANCE ? 'text-red-500' : 'text-green-400'}`}>{detectedObj.dist.toFixed(1)}m</span>
                 </div>
               )}
               {!isDetecting && (
                 <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center">
                   <span className="text-xl font-bold text-yellow-400 tracking-widest border-2 border-yellow-400/50 px-4 py-2 rounded-lg bg-black/50">
                     PAUSED
                   </span>
                 </div>
               )}
             </div>
             <div className="bg-gray-800 p-3 rounded text-xs flex items-center gap-4 border border-gray-700 mt-auto">
               <span className="font-bold text-gray-500">CALIBRATE</span>
               <input type="range" min="400" max="1500" value={focalLength} onChange={e => setFocalLength(Number(e.target.value))} className="flex-1 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer" />
               <span className="font-mono w-10">{focalLength}</span>
             </div>
          </div>
          </div>
        </main>
      )}

      <footer className="bg-gray-800 p-2 text-[10px] flex justify-between items-center border-t border-gray-700">
        <button onClick={() => logger.downloadCSV()} className="bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded">Export Logs</button>
        <span className="text-gray-600 italic">v2.1 • TensorFlow.js WebGL • Multi-Threaded Simulation</span>
      </footer>
    </div>
  );
};

createRoot(document.getElementById('root')!).render(<App />);
