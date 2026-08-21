/* ════════════════════════════════════════════════════════════
   Voice RAG — Canvas Wave Engine v10
   CSS glass orb handles the sphere look.
   Canvas draws: background, wide outer waves, simple inner
   waves (visible through the glass), sparkle dots, particles.
   ════════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {

    /* ── DOM ── */
    const recordBtn    = document.getElementById("id-record-btn");
    const glowOrb      = document.getElementById("id-glowing-orb");
    const statusLabel  = document.getElementById("id-recording-status");
    const convBox      = document.getElementById("id-conversation-container");
    const transText    = document.getElementById("id-transcription-text");
    const answerText   = document.getElementById("id-answer-text");
    const textDrawer   = document.getElementById("id-text-input-drawer");
    const toggleTxtBtn = document.getElementById("id-toggle-text-drawer-btn");
    const textInput    = document.getElementById("id-text-input");
    const submitBtn    = document.getElementById("id-submit-text-btn");
    const menuBtn      = document.getElementById("id-menu-toggle-btn");
    const openDrBtn    = document.getElementById("id-open-drawer-btn");
    const closeDrBtn   = document.getElementById("id-close-drawer-btn");
    const saveDrBtn    = document.getElementById("id-save-drawer-btn");
    const drawer       = document.getElementById("id-sidebar-drawer");
    const backdrop     = document.getElementById("id-drawer-backdrop");
    const sttSel       = document.getElementById("id-stt-provider-select");
    const chunkSel     = document.getElementById("id-chunking-strategy-select");
    const sarvamKey    = document.getElementById("id-sarvam-key-input");
    const elevenKey    = document.getElementById("id-elevenlabs-key-input");
    const totalMsEl    = document.getElementById("id-total-ms-val");
    const chunksListEl = null; // section removed
    const benchmarkBtn = document.getElementById("id-run-benchmark-btn");
    const p50El        = document.getElementById("id-p50-val");
    const p70El        = document.getElementById("id-p70-val");
    const p90El        = document.getElementById("id-p90-val");
    const p100El       = document.getElementById("id-p100-val");

    /* ── Canvas ── */
    const canvas = document.getElementById("id-orb-canvas");
    const ctx    = canvas.getContext("2d");
    const stage  = document.getElementById("id-assistant-stage");

    let W, H, cx, cy, R;
    function resize() {
        W  = canvas.width  = stage.clientWidth  || window.innerWidth;
        H  = canvas.height = stage.clientHeight || window.innerHeight;
        cx = W / 2; cy = H / 2;
        // Sync with CSS orb (300px / 2 = 150px)
        R  = Math.min(W, H) * 0.155;
        R  = Math.max(110, Math.min(R, 152));
    }
    window.addEventListener("resize", resize);
    resize();

    /* ── State ── */
    let energy      = 0;
    let targetE     = 0;
    let t           = 0;
    let isRecording = false;
    let analyserNode = null, audioCtx = null, mediaRecorder = null, freqData = null;

    /* ── Particles ── */
    const PART_COUNT = 24;
    const parts = Array.from({ length: PART_COUNT }, (_, i) => ({
        angle: (i / PART_COUNT) * Math.PI * 2 + Math.random() * 0.3,
        orbit: R * (1.25 + Math.random() * 0.65),
        speed: (Math.random() > 0.5 ? 1 : -1) * (0.00022 + Math.random() * 0.00035),
        size:  0.7 + Math.random() * 1.4,
        alpha: 0.15 + Math.random() * 0.45,
        pp:    Math.random() * Math.PI * 2,
    }));

    /* ══════════════════════════════════════
       RENDER LOOP
    ══════════════════════════════════════ */
    function draw() {
        t += 0.014;
        energy += (targetE - energy) * 0.055;

        ctx.clearRect(0, 0, W, H);
        drawBg();
        drawWaves();      // continuous full-width ribbons through the orb
        drawParticles();

        requestAnimationFrame(draw);
    }
    draw();

    /* ────────────────────────────────────────────────────
       1. BACKGROUND
    ──────────────────────────────────────────────────── */
    function drawBg() {
        ctx.fillStyle = "#030408";  // slightly darker
        ctx.fillRect(0, 0, W, H);

        const atm = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 3.8);
        atm.addColorStop(0,   `rgba(55,20,145, ${0.30 + energy * 0.22})`);
        atm.addColorStop(0.48,`rgba(24,8,72,   ${0.13 + energy * 0.08})`);
        atm.addColorStop(1,   "rgba(3,4,10,0)");
        ctx.fillStyle = atm;
        ctx.fillRect(0, 0, W, H);

        const vig = ctx.createRadialGradient(cx, cy, R * 0.4, cx, cy, Math.hypot(W,H) * 0.68);
        vig.addColorStop(0, "rgba(0,0,0,0)");
        vig.addColorStop(1, "rgba(0,0,0,0.72)");
        ctx.fillStyle = vig;
        ctx.fillRect(0, 0, W, H);
    }

    /* ────────────────────────────────────────────────────
       2. CONTINUOUS FULL-WIDTH WAVES
          Single unbroken ribbon from left edge to right edge.
          The CSS glass orb (dark, semi-transparent) sits on top
          so the waves are naturally dimmed as they pass through
          the sphere — creating perfect seamless continuity.
          Left portion: purple-violet | Right: electric blue
    ──────────────────────────────────────────────────── */
    function drawWaves() {
        const LAYERS = 28;
        const AMP    = R * (0.78 + energy * 0.70);

        // Build a full-width purple→blue gradient (reused for all layers)
        // Each layer has same shape; colour is determined by x position.
        for (let layer = 0; layer < LAYERS; layer++) {
            const norm  = layer / (LAYERS - 1);
            const yOff  = (norm - 0.5) * R * 0.60;
            const phase = layer * 0.22 + Math.sin(layer * 0.82) * 0.45;

            // Middle layers are brightest (silk-ribbon look)
            const bandAlpha = 0.055 + 0.60 * Math.sin(norm * Math.PI);
            const lw = (norm > 0.26 && norm < 0.74) ? 1.75 : 0.75;

            // Full-width gradient: purple left → near-invisible center → blue right
            // The orb's dark glass body handles the visual separation in the middle
            const grad = ctx.createLinearGradient(0, 0, W, 0);
            grad.addColorStop(0,    "rgba(75,12,195, 0.0)");
            grad.addColorStop(0.08, `rgba(112,38,238, ${bandAlpha * 0.70})`);
            grad.addColorStop(0.30, `rgba(148,58,255, ${bandAlpha})`);
            grad.addColorStop(0.44, `rgba(120,50,255, ${bandAlpha * 0.55})`);
            grad.addColorStop(0.50, `rgba(80, 60,255, ${bandAlpha * 0.28})`);
            grad.addColorStop(0.56, `rgba(38,120,255, ${bandAlpha * 0.55})`);
            grad.addColorStop(0.70, `rgba(22,158,255, ${bandAlpha})`);
            grad.addColorStop(0.92, `rgba(15,142,240, ${bandAlpha * 0.70})`);
            grad.addColorStop(1,    "rgba(8,118,215, 0.0)");

            ctx.beginPath();
            ctx.lineWidth   = lw;
            ctx.strokeStyle = grad;

            for (let x = 0; x <= W; x += 2) {
                // Bell envelope — zero at screen edges, peak at centre
                const env = Math.sin((x / W) * Math.PI);
                const y   = cy + yOff
                    + Math.sin(x * 0.0072 + t * 0.80 + phase)       * AMP * env
                    + Math.sin(x * 0.015  - t * 0.56 + phase * 1.3) * AMP * 0.44 * env
                    + Math.sin(x * 0.031  + t * 0.32)                * AMP * 0.16 * env;
                if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Sparkle dot columns on both sides of orb
        drawSparkles(0, cx - R * 1.22, 1);
        drawSparkles(cx + R * 1.22, W, 2);
    }

    function drawSparkles(xStart, xEnd, side) {
        const COLS = 65;
        for (let i = 0; i < COLS; i++) {
            const x   = xStart + (i / COLS) * (xEnd - xStart);
            const nx  = x / W;
            const env = Math.sin(nx * Math.PI);
            const maxH = env * (isRecording ? 46 : 30) * (0.30 + 0.70 * (Math.sin(i * 0.68 + t * 2.1) * 0.5 + 0.5));
            if (maxH < 2) continue;
            ctx.fillStyle = side === 1 ? "rgba(148,58,255,0.42)" : "rgba(22,158,255,0.42)";
            for (let dy = -maxH; dy <= maxH; dy += 6) {
                ctx.globalAlpha = (1 - Math.abs(dy) / maxH) * 0.80;
                ctx.fillRect(x, cy + dy, 1.5, 1.5);
            }
        }
        ctx.globalAlpha = 1;
    }

    /* ────────────────────────────────────────────────────
       4. PARTICLES — orbiting ellipse around orb
    ──────────────────────────────────────────────────── */
    function drawParticles() {
        parts.forEach(p => {
            p.angle += p.speed * (1 + energy * 2.5);
            const orbit = p.orbit * (1 + energy * 0.14);
            const px    = cx + Math.cos(p.angle) * orbit;
            const py    = cy + Math.sin(p.angle) * orbit * 0.52;

            const pulse = 0.5 + 0.5 * Math.sin(t * 1.8 + p.pp);
            const alpha = p.alpha * (0.38 + pulse * 0.62) * (0.42 + energy * 0.58);
            const sz    = p.size * (0.65 + pulse * 0.35 + energy * 0.45);

            ctx.beginPath();
            ctx.arc(px, py, sz, 0, Math.PI * 2);
            ctx.fillStyle = Math.cos(p.angle) < 0
                ? `rgba(148,58,255,${alpha})`
                : `rgba(22,158,255,${alpha})`;
            ctx.fill();
        });
    }

    /* ══════════════════════════════════════
       SPEECH RECOGNITION
    ══════════════════════════════════════ */
    let speechRec = null;
    const SRAPI   = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SRAPI) {
        speechRec = new SRAPI();
        speechRec.continuous = false; speechRec.interimResults = true; speechRec.lang = "en-US";
        speechRec.onresult = ev => {
            let tr = "";
            for (let i = ev.resultIndex; i < ev.results.length; i++) tr += ev.results[i][0].transcript;
            if (tr.trim()) { textInput.value = tr.trim(); statusLabel.textContent = `"${tr.trim()}"`; }
        };
        speechRec.onend = () => { const q = textInput.value.trim(); if (isRecording) { stopRec(); if (q) runRAG(null, q); } };
    }

    async function startRec() {
        isRecording = true; targetE = 0.68;
        recordBtn.classList.add("recording");
        glowOrb?.classList.add("active-recording");
        statusLabel.textContent = "Listening… speak now";
        textInput.value = "";
        // Clear previous Q&A before new question
        convBox.classList.add("hidden");
        transText.innerText = "";
        answerText.innerText = "";
        spawnRipple();
        try { speechRec?.start(); } catch(_) {}
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioCtx      = new (window.AudioContext || window.webkitAudioContext)();
            analyserNode  = audioCtx.createAnalyser(); analyserNode.fftSize = 256;
            freqData      = new Uint8Array(analyserNode.frequencyBinCount);
            audioCtx.createMediaStreamSource(stream).connect(analyserNode);
            mediaRecorder = new MediaRecorder(stream); mediaRecorder.start();
            pumpMic();
        } catch(e) { console.warn("Mic:", e.message); }
    }

    function stopRec() {
        isRecording = false; targetE = 0;
        recordBtn.classList.remove("recording");
        glowOrb?.classList.remove("active-recording");
        statusLabel.textContent = "Tap the microphone to speak";
        try { speechRec?.stop(); } catch(_) {}
        if (mediaRecorder?.state !== "inactive") mediaRecorder?.stop();
    }

    function pumpMic() {
        if (!isRecording || !analyserNode) return;
        analyserNode.getByteFrequencyData(freqData);
        let sum = 0; for (const v of freqData) sum += v;
        targetE = Math.min(1, (sum / freqData.length) / 85);
        requestAnimationFrame(pumpMic);
    }

    function spawnRipple() {
        const rect = recordBtn.getBoundingClientRect();
        const el   = document.createElement("div");
        el.className = "mic-ripple";
        Object.assign(el.style, { left:`${rect.left}px`, top:`${rect.top}px`, width:`${rect.width}px`, height:`${rect.height}px` });
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 750);
    }

    recordBtn?.addEventListener("click", e => { e.preventDefault(); isRecording ? stopRec() : startRec(); });

    /* ── Drawer ── */
    const openDr  = () => { drawer.classList.remove("hidden"); backdrop.classList.remove("hidden"); };
    const closeDr = () => { drawer.classList.add("hidden"); backdrop.classList.add("hidden"); };
    menuBtn  ?.addEventListener("click", openDr);
    openDrBtn?.addEventListener("click", openDr);
    closeDrBtn?.addEventListener("click", closeDr);
    saveDrBtn ?.addEventListener("click", closeDr);
    backdrop  ?.addEventListener("click", closeDr);

    toggleTxtBtn?.addEventListener("click", () => {
        textDrawer.classList.toggle("hidden");
        if (!textDrawer.classList.contains("hidden")) textInput.focus();
    });
    submitBtn?.addEventListener("click", e => { e.preventDefault(); const q=textInput.value.trim(); if(q) runRAG(null,q); });

    /* ══════════════════════════════════════
       RENDER RESULTS
    ══════════════════════════════════════ */
    function renderResults(data) {
        // Show conversation
        convBox.classList.remove("hidden");
        transText.innerText  = `"${data.transcription}"`;
        answerText.innerText  = data.answer;
        answerText.classList.remove("highlight-pulse");
        void answerText.offsetWidth;
        answerText.classList.add("highlight-pulse");

        // Latency badge
        totalMsEl.innerText = data.timing.total_latency_ms;

        // Step times and nodes
        const stepIds    = ["id-step-stt","id-step-guard-in","id-step-vector","id-step-gen","id-step-guard-out"];
        const timeEls    = ["id-time-stt","id-time-guard-in","id-time-vector","id-time-gen","id-time-guard-out"];
        const timeValues = [
            data.timing.stt_ms,
            data.timing.guardrail_input_ms,
            data.timing.vector_retrieval_ms,
            data.timing.generation_ms,
            data.timing.guardrail_output_ms,
        ];

        // Reset
        stepIds.forEach(id => document.getElementById(id)?.classList.remove("active"));
        for (let i=0; i<5; i++) document.getElementById(`td-${i}`)?.classList.remove("done");
        const fill = document.getElementById("id-step-fill");
        if (fill) fill.style.width = "0%";

        // Animate steps one-by-one
        stepIds.forEach((id, idx) => {
            setTimeout(() => {
                document.getElementById(id)?.classList.add("active");
                const tel = document.getElementById(timeEls[idx]);
                if (tel) tel.innerText = `${timeValues[idx]}ms`;
                document.getElementById(`td-${idx}`)?.classList.add("done");
                if (fill) fill.style.width = `${((idx + 1) / 5) * 100}%`;
            }, idx * 200);
        });

    }

    /* ══════════════════════════════════════
       POLAR ORBIT RADAR CHART
    ══════════════════════════════════════ */
    function initRadarChart(p50,p70,p90,p100) {
        const canvas = document.getElementById("id-radar-canvas");
        if (!canvas) return;
        const ctx2 = canvas.getContext("2d");
        const size  = canvas.width;   // 300
        const cx2   = size / 2, cy2 = size / 2;
        const maxR  = 115;
        const maxMs = Math.max(p50,p70,p90,p100) * 1.18;

        const dots = [
            { label:"P50",  ms:p50,  color:"#22d3ee", angle: 3.67  }, // ~210°
            { label:"P70",  ms:p70,  color:"#4ade80", angle:-1.05  }, // ~300° (top-right)
            { label:"P90",  ms:p90,  color:"#fb923c", angle: 0.17  }, // ~10°
            { label:"P100", ms:p100, color:"#f87171", angle: 1.75  }, // ~100°
        ];

        ctx2.clearRect(0,0,size,size);

        // Background radial glow
        const bg = ctx2.createRadialGradient(cx2,cy2,0,cx2,cy2,maxR*1.3);
        bg.addColorStop(0,  "rgba(100,55,230,0.14)");
        bg.addColorStop(1,  "rgba(0,0,0,0)");
        ctx2.fillStyle = bg;
        ctx2.fillRect(0,0,size,size);

        // Draw orbit circles + dots
        dots.forEach(d => {
            const r = (d.ms / maxMs) * maxR;
            const dx = cx2 + r * Math.cos(d.angle);
            const dy = cy2 + r * Math.sin(d.angle);

            // Dashed orbit ring
            ctx2.beginPath();
            ctx2.arc(cx2, cy2, r, 0, Math.PI*2);
            ctx2.strokeStyle = d.color + "28";
            ctx2.lineWidth = 1;
            ctx2.setLineDash([4,7]);
            ctx2.stroke();
            ctx2.setLineDash([]);

            // Glow halo around dot
            const glow = ctx2.createRadialGradient(dx,dy,0,dx,dy,22);
            glow.addColorStop(0, d.color + "88");
            glow.addColorStop(1, "transparent");
            ctx2.beginPath();
            ctx2.arc(dx, dy, 22, 0, Math.PI*2);
            ctx2.fillStyle = glow; ctx2.fill();

            // Dot
            ctx2.beginPath();
            ctx2.arc(dx, dy, 6, 0, Math.PI*2);
            ctx2.fillStyle = d.color; ctx2.fill();
            ctx2.strokeStyle = "rgba(255,255,255,0.6)";
            ctx2.lineWidth = 1.5; ctx2.stroke();

            // Label
            const isLeft = dx < cx2;
            const lx = dx + (isLeft ? -10 : 10);
            const ly = dy + (dy < cy2 ? -14 : 18);
            ctx2.textAlign = isLeft ? "right" : "left";
            ctx2.fillStyle = "rgba(255,255,255,0.82)";
            ctx2.font = "600 11px Outfit,sans-serif";
            ctx2.fillText(d.label, lx, ly);
            ctx2.fillStyle = d.color;
            ctx2.font = "500 10px JetBrains Mono,monospace";
            ctx2.fillText(d.ms + " ms", lx, ly + 13);
        });

        // Centre sphere
        const cR    = 38;
        const cGrad = ctx2.createRadialGradient(cx2-8,cy2-10,0,cx2,cy2,cR);
        cGrad.addColorStop(0,  "rgba(200,170,255,0.65)");
        cGrad.addColorStop(0.4,"rgba(100,55,230,0.45)");
        cGrad.addColorStop(1,  "rgba(20,10,55,0.95)");
        ctx2.beginPath();
        ctx2.arc(cx2,cy2,cR,0,Math.PI*2);
        ctx2.fillStyle = cGrad; ctx2.fill();
        ctx2.strokeStyle = "rgba(180,140,255,0.55)";
        ctx2.lineWidth = 1.5; ctx2.stroke();

        // "ms" text
        ctx2.fillStyle = "rgba(255,255,255,0.88)";
        ctx2.font = "bold 15px Outfit,sans-serif";
        ctx2.textAlign = "center";
        ctx2.textBaseline = "middle";
        ctx2.fillText("ms", cx2, cy2);
        ctx2.textBaseline = "alphabetic";
    }

    /* ── Sparklines ── */
    function buildSparkline(id, seed) {
        const el = document.getElementById(id);
        if (!el) return;
        const pts = [];
        for (let i=0; i<=18; i++) {
            const x = (i/18)*80;
            const noise = Math.sin(i*0.82+seed)*6 + Math.sin(i*1.6+seed*1.8)*3.5;
            pts.push(`${x.toFixed(1)},${Math.max(2,Math.min(28, 15 - noise)).toFixed(1)}`);
        }
        el.setAttribute("points", pts.join(" "));
    }

    // Init radar + sparklines on load
    initRadarChart(18.15, 19.33, 24.23, 38.45);
    buildSparkline("sp-p50",  1.2);
    buildSparkline("sp-p70",  2.5);
    buildSparkline("sp-p90",  4.1);
    buildSparkline("sp-p100", 6.8);

    /* ══════════════════════════════════════
       RAG PIPELINE
    ══════════════════════════════════════ */
    async function runRAG(audioBlob, query) {
        targetE = 0.40; statusLabel.textContent = "⚡ Synthesizing answer…";
        document.querySelectorAll(".step-time").forEach(el => el.innerText = "…");

        const fd = new FormData();
        if (audioBlob) fd.append("audio", audioBlob, "rec.wav");
        if (query)     fd.append("query_text", query);
        fd.append("stt_provider", sttSel.value);
        fd.append("chunking_strategy", chunkSel.value);
        if (sarvamKey?.value)  fd.append("sarvam_key",     sarvamKey.value);
        if (elevenKey?.value)  fd.append("elevenlabs_key", elevenKey.value);

        try {
            const res = await fetch("/api/voice-rag", { method:"POST", body:fd });
            if (!res.ok) {
                let errText = "HTTP " + res.status;
                try {
                    const jsonErr = await res.json();
                    if (jsonErr && jsonErr.detail) errText = jsonErr.detail;
                } catch(_) {}
                throw new Error(errText);
            }
            renderResults(await res.json());
        } catch(err) {
            convBox.classList.remove("hidden");
            answerText.innerHTML = `<span style="color:#f87171">Error: ${err.message}</span>`;
        } finally { targetE = 0; statusLabel.textContent = "Tap the microphone to speak"; }
    }

    /* ── Benchmark ── */
    benchmarkBtn?.addEventListener("click", async e => {
        e.preventDefault(); benchmarkBtn.disabled=true; benchmarkBtn.innerText="⏳ Running…";
        try {
            const res=await fetch("/api/benchmark",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({chunking_strategy:chunkSel.value,stt_provider:sttSel.value})});
            if(!res.ok) throw new Error();
            const m=(await res.json()).latency_metrics;
            p50El.innerHTML=`${m.p50_ms} <small>ms</small>`;
            p70El.innerHTML=`${m.p70_ms} <small>ms</small>`;
            p90El.innerHTML=`${m.p90_ms} <small>ms</small>`;
            p100El.innerHTML=`${m.p100_ms} <small>ms</small>`;
        } catch{ alert("Benchmark failed"); }
        finally{ benchmarkBtn.disabled=false; benchmarkBtn.innerText="⚡ Run Benchmark"; }
    });
});
