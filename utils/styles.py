"""
utils/styles.py — Project Sambhav UI System
Section 24 — Design tokens, nav, disclaimer
"""

# ── SVG Logo (Section 21 — two interlocking hexagons + S letterform) ──────────
LOGO_SVG = """<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" width="32" height="32">
  <polygon points="20,2 30,8 30,20 20,26 10,20 10,8" fill="none" stroke="#C2CD93" stroke-width="1.5"/>
  <polygon points="20,14 30,20 30,32 20,38 10,32 10,20" fill="none" stroke="#C891AA" stroke-width="1.2" opacity="0.7"/>
  <text x="20" y="23" text-anchor="middle" font-family="Poppins,sans-serif"
        font-weight="800" font-size="13" fill="#C2CD93">S</text>
</svg>"""

LOGO_DATA_URI = "data:image/svg+xml;charset=utf-8," + LOGO_SVG.replace("\n","").replace("  ","")


def load_css():
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:             #08080A;
    --surface:        #141419;
    --surface-2:      #1A1A21;
    --input-bg:       #0D0D10;
    --accent:         #C2CD93;
    --accent-dim:     #788258;
    --accent-fade:    #4B5234;
    --sakura:         #C891AA;
    --sakura-deep:    #9B5A78;
    --text-primary:   #EBE9F2;
    --text-secondary: #73717D;
    --text-muted:     #34323C;
    --border:         #26242E;
    --border-2:       #373441;
}

/* ── STREAMLIT CLEANUP ───────────────────────── */
.stApp { background: var(--bg) !important; font-family: 'Poppins', sans-serif !important; }
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; visibility: hidden !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"],
section[data-testid="stSidebarContent"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.main .block-container { padding: 0 40px !important; max-width: 100% !important; }

/* ── BACKGROUND ATMOSPHERE ───────────────────── */
body::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 600px 300px at 50% 0%, rgba(155,90,120,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 800px 600px at 50% 40%, rgba(19,19,24,0.9) 0%, #08080A 100%);
    pointer-events: none; z-index: 0;
}

/* ── NAV BAR ─────────────────────────────────── */
.s-nav {
    position: fixed; top: 0; left: 0; right: 0; height: 64px;
    background: rgba(8,8,10,0.88);
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px; z-index: 9000;
}
.nav-logo { display: flex; align-items: center; gap: 10px; cursor: pointer; text-decoration: none; }
.nav-logo svg { width: 32px; height: 32px; }
.nav-wordmark {
    font-family: 'Poppins', sans-serif; font-weight: 700;
    font-size: 14px; letter-spacing: 0.12em; color: var(--text-primary);
}
.nav-links { display: flex; align-items: center; gap: 36px; }
.nav-link {
    font-family: 'Poppins', sans-serif; font-weight: 500;
    font-size: 13px; color: var(--text-secondary);
    cursor: pointer; position: relative; padding-bottom: 6px;
    transition: color 0.2s ease; text-decoration: none; white-space: nowrap;
    background: none; border: none;
}
.nav-link::after {
    content: ''; position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
    width: 0; height: 4px; border-radius: 50%; background: var(--accent-dim);
    transition: width 0.2s ease;
}
.nav-link:hover { color: var(--text-primary); }
.nav-link:hover::after, .nav-link.active::after { width: 4px; }
.nav-link.active { color: var(--text-primary); }

.nav-right { display: flex; align-items: center; gap: 14px; }
.nav-badge { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); }

/* ── HAMBURGER ───────────────────────────────── */
.s-hamburger {
    display: flex; flex-direction: column; gap: 5px;
    cursor: pointer; padding: 8px; z-index: 10000; position: relative;
    background: none; border: none;
}
.s-hamburger span {
    display: block; width: 24px; height: 1.5px; background: var(--text-primary);
    transition: all 0.4s cubic-bezier(0.77,0,0.175,1); transform-origin: center;
}
.s-hamburger.open span:nth-child(1) { transform: translateY(6.5px) rotate(45deg); }
.s-hamburger.open span:nth-child(2) { opacity: 0; transform: scaleX(0); }
.s-hamburger.open span:nth-child(3) { transform: translateY(-6.5px) rotate(-45deg); }

/* ── BUTTONS ─────────────────────────────────── */
.btn-ghost {
    background: transparent; border: 1px solid var(--border);
    color: var(--text-secondary); font-family: 'Poppins', sans-serif;
    font-weight: 500; font-size: 13px; padding: 7px 18px;
    border-radius: 8px; cursor: pointer; transition: all 0.15s ease;
}
.btn-ghost:hover { border-color: var(--border-2); color: var(--text-primary); }

.btn-sakura {
    background: #130D16; border: 1px solid var(--sakura-deep); color: var(--sakura);
    font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 13px;
    padding: 9px 20px; border-radius: 8px; cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 0 12px rgba(155,90,120,0.15);
}
.btn-sakura:hover {
    border-color: var(--sakura);
    box-shadow: 0 0 24px rgba(200,145,170,0.3);
    transform: translateY(-1px);
}

.btn-primary {
    background: #130D16; border: 1px solid var(--sakura-deep); color: var(--sakura);
    font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 15px;
    padding: 14px 32px; border-radius: 10px; cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 0 20px rgba(155,90,120,0.2);
}
.btn-primary:hover {
    border-color: var(--sakura);
    box-shadow: 0 0 32px rgba(200,145,170,0.35);
    transform: translateY(-2px);
}

.btn-secondary-glow {
    background: #0D100A; border: 1px solid var(--accent-dim); color: var(--accent);
    font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 14px;
    padding: 12px 28px; border-radius: 10px; cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 0 20px rgba(194,205,147,0.15);
}
.btn-secondary-glow:hover {
    border-color: var(--accent);
    box-shadow: 0 0 32px rgba(194,205,147,0.3);
    transform: translateY(-2px);
}

.btn-secondary {
    background: var(--input-bg); border: 1px solid var(--border);
    color: var(--text-secondary); font-family: 'Poppins', sans-serif;
    font-size: 15px; padding: 14px 32px; border-radius: 10px; cursor: pointer;
    transition: all 0.2s ease;
}
.btn-secondary:hover { border-color: var(--border-2); color: var(--text-primary); }

/* ── FULLSCREEN MENU — STARTS CLOSED ─────────── */
.s-menu-overlay {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: #08080A; z-index: 9500;
    display: flex; flex-direction: column;
    justify-content: center; align-items: flex-start; padding: 0 10vw;
    clip-path: inset(0 0 100% 0);
    transition: clip-path 0.7s cubic-bezier(0.77,0,0.175,1);
    pointer-events: none;
}
.s-menu-overlay.open {
    clip-path: inset(0 0 0% 0);
    pointer-events: all;
}
.s-menu-item {
    font-family: 'Poppins', sans-serif;
    font-size: clamp(42px,6vw,80px); font-weight: 800; color: #EBE9F2;
    display: block; line-height: 1.1; margin-bottom: 12px;
    opacity: 0; transform: translateY(40px);
    transition: opacity 0.5s ease, transform 0.5s ease, color 0.2s ease;
    cursor: pointer; background: none; border: none; text-align: left;
}
.s-menu-overlay.open .s-menu-item:nth-child(1){opacity:1;transform:translateY(0);transition-delay:0.15s;}
.s-menu-overlay.open .s-menu-item:nth-child(2){opacity:1;transform:translateY(0);transition-delay:0.22s;}
.s-menu-overlay.open .s-menu-item:nth-child(3){opacity:1;transform:translateY(0);transition-delay:0.29s;}
.s-menu-overlay.open .s-menu-item:nth-child(4){opacity:1;transform:translateY(0);transition-delay:0.36s;}
.s-menu-overlay.open .s-menu-item:nth-child(5){opacity:1;transform:translateY(0);transition-delay:0.43s;}
.s-menu-overlay.open .s-menu-item:nth-child(6){opacity:1;transform:translateY(0);transition-delay:0.50s;}
.s-menu-item:hover { color: var(--accent); }
.s-menu-item.cta-item { color: var(--sakura); }
.s-menu-item.cta-item:hover { color: #EBE9F2; }
.s-menu-line { width:100%; height:1px; background:var(--border); margin:24px 0; opacity:0; transition:opacity 0.4s ease 0.2s; }
.s-menu-overlay.open .s-menu-line { opacity:1; }
.s-menu-sub { display:flex; gap:40px; margin-top:8px; opacity:0; transform:translateY(20px); transition:opacity 0.5s ease 0.55s, transform 0.5s ease 0.55s; }
.s-menu-overlay.open .s-menu-sub { opacity:1; transform:translateY(0); }
.s-menu-sub-item { font-family:'Poppins',sans-serif; font-size:13px; color:var(--text-secondary); cursor:pointer; }
.s-menu-sub-item:hover { color:#EBE9F2; }

/* ── SWIPE STRIP ─────────────────────────────── */
.swipe-strip { display:flex; gap:20px; overflow:hidden; cursor:grab; user-select:none; }
.swipe-strip:active { cursor:grabbing; }
.swipe-strip-inner { display:flex; gap:20px; transition:transform 0.6s cubic-bezier(0.25,0.46,0.45,0.94); will-change:transform; }
.swipe-card {
    flex:0 0 280px; height:180px; background:var(--surface);
    border:1px solid var(--border); border-radius:12px;
    display:flex; flex-direction:column; justify-content:flex-end; padding:20px;
    position:relative; overflow:hidden; transition:border-color 0.3s, transform 0.3s;
}
.swipe-card:hover { border-color:var(--accent-dim); transform:translateY(-4px); }
.swipe-card-num { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-muted); position:absolute; top:16px; right:16px; }
.swipe-card-title { font-family:'Poppins',sans-serif; font-size:15px; font-weight:700; color:var(--text-primary); margin-bottom:4px; }
.swipe-card-desc { font-family:'Poppins',sans-serif; font-size:11px; color:var(--text-secondary); }

/* ── PAGE TRANSITION ─────────────────────────── */
.page-transition { position:fixed; top:0; left:0; width:100vw; height:100vh; background:#08080A; z-index:8888; pointer-events:none; opacity:0; transition:opacity 0.3s ease; }
.page-transition.active { opacity:1; }

/* ── MODE CHIPS ──────────────────────────────── */
.chip {
    display:inline-flex; align-items:center;
    background:var(--input-bg); border:1px solid var(--border);
    color:var(--text-muted); font-family:'Poppins',sans-serif;
    font-size:13px; padding:5px 16px; border-radius:100px;
    cursor:pointer; transition:all 0.2s ease; white-space:nowrap;
}
.chip:hover { border-color:var(--border-2); color:var(--text-secondary); }
.chip.active { background:rgba(18,20,26,0.9); border-color:var(--accent-dim); color:var(--accent); }

/* ── EYEBROW PILL ────────────────────────────── */
.eyebrow-pill { display:inline-flex; align-items:center; gap:8px; background:var(--surface); border:1px solid var(--border); border-radius:100px; padding:6px 16px; margin-bottom:32px; }
.eyebrow-dot { width:6px; height:6px; border-radius:50%; background:var(--accent); flex-shrink:0; }
.eyebrow-text { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-secondary); letter-spacing:0.08em; }

/* ── HERO TYPOGRAPHY ─────────────────────────── */
.hero-h1 { font-family:'Poppins',sans-serif; font-weight:800; font-size:clamp(56px,7vw,96px); line-height:0.95; color:var(--text-primary); margin-bottom:4px; animation:fadeUp 0.7s ease forwards; }
.hero-h2 { font-family:'Poppins',sans-serif; font-weight:800; font-size:clamp(48px,6vw,72px); line-height:1.0; color:var(--text-primary); margin-bottom:28px; animation:fadeUp 0.8s ease forwards; }
.hero-accent { color:var(--accent); }
.hero-desc { font-family:'Poppins',sans-serif; font-weight:400; font-size:17px; color:var(--text-secondary); line-height:1.65; max-width:520px; margin:0 auto 36px; animation:fadeUp 0.9s ease forwards; }

/* ── SIDE STATS ──────────────────────────────── */
.side-stat-val { font-family:'Poppins',sans-serif; font-weight:800; font-size:48px; color:var(--text-primary); line-height:1; }
.side-stat-mono { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:36px; color:var(--text-primary); line-height:1; }
.side-stat-lbl { font-family:'Poppins',sans-serif; font-size:13px; color:var(--text-secondary); margin-top:4px; }

/* ── STATS STRIP ─────────────────────────────── */
.stats-strip { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--border); border-radius:14px; overflow:hidden; margin:40px 0; }
.stat-cell { padding:28px 32px; border-right:1px solid var(--border); transition:background 0.2s; }
.stat-cell:last-child { border-right:none; }
.stat-cell:hover { background:var(--surface); }
.stat-val { font-family:'JetBrains Mono',monospace; font-weight:500; font-size:28px; color:var(--accent); }
.stat-lbl { font-family:'Poppins',sans-serif; font-size:13px; color:var(--text-secondary); margin-top:4px; }

/* ── FEATURE ROWS ────────────────────────────── */
.feature-row { display:flex; align-items:flex-start; padding:28px 0; border-bottom:1px solid var(--border); }
.feature-row:hover .feature-title { color:var(--accent); }
.feature-num { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--accent-dim); min-width:36px; padding-top:3px; }
.feature-title { font-family:'Poppins',sans-serif; font-weight:700; font-size:15px; color:var(--text-primary); margin-bottom:6px; transition:color 0.2s; }
.feature-desc { font-family:'Poppins',sans-serif; font-size:13px; color:var(--text-secondary); line-height:1.6; }

/* ── DISCLAIMER BAR ──────────────────────────── */
.disclaimer-bar {
    position:fixed; bottom:0; left:0; right:0; height:28px;
    background:#0A090D; border-top:1px solid var(--border);
    display:flex; align-items:center; justify-content:center; z-index:9000;
}
.disclaimer-txt { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--text-muted); }

/* ── ANIMATIONS ──────────────────────────────── */
@keyframes fadeUp { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }

/* ── SCROLLBAR ───────────────────────────────── */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border-2); border-radius:2px; }

/* ── PAGE WRAP ───────────────────────────────── */
.page-wrap { padding-top:80px; padding-bottom:40px; position:relative; z-index:1; }
</style>
"""


def nav_html(active=""):
    from utils.styles import LOGO_SVG
    pages = [
        ("Modes",      "2_Modes"),
        ("Dashboard",  "3_Dashboard"),
        ("Fact-Check", "4_Fact_Check"),
        ("Privacy",    "5_Privacy"),
    ]
    links = ""
    for name, _ in pages:
        cls = "nav-link active" if active == name else "nav-link"
        links += f'<button class="{cls}" onclick="sambhavNav(\'{name}\')">{name}</button>'

    return f"""
<!-- PAGE TRANSITION -->
<div class="page-transition" id="pageTransition"></div>

<!-- FULLSCREEN MENU — starts closed (no .open class) -->
<div class="s-menu-overlay" id="menuOverlay">
  <button class="s-menu-item" onclick="sambhavNav('Dashboard')">Dashboard</button>
  <button class="s-menu-item" onclick="sambhavNav('Modes')">Modes</button>
  <button class="s-menu-item" onclick="sambhavNav('Fact-Check')">Fact&#8209;Check</button>
  <button class="s-menu-item" onclick="sambhavNav('Privacy')">Privacy</button>
  <div class="s-menu-line"></div>
  <button class="s-menu-item cta-item" onclick="sambhavNav('Dashboard')">Try Sambhav</button>
  <div class="s-menu-sub">
    <span class="s-menu-sub-item">Academic v1.0</span>
    <span class="s-menu-sub-item">Sri Indu Institute</span>
    <span class="s-menu-sub-item">2025&#8209;2026</span>
  </div>
</div>

<!-- NAV BAR -->
<div class="s-nav">
  <div class="nav-logo">
    {LOGO_SVG}
    <span class="nav-wordmark">SAMBHAV</span>
  </div>
  <div class="nav-links">{links}</div>
  <div class="nav-right">
    <span class="nav-badge">Academic v1.0</span>
    <button class="btn-ghost" onclick="sambhavNav('Login')">Sign in</button>
    <button class="btn-sakura" onclick="sambhavNav('Dashboard')">Try Now</button>
    <button class="s-hamburger" id="hamburger" onclick="toggleMenu()" aria-label="Menu">
      <span></span><span></span><span></span>
    </button>
  </div>
</div>

<script>
// ── Menu toggle — no auto-open ─────────────────
var menuOpen = false;
function toggleMenu() {{
  menuOpen = !menuOpen;
  var overlay = document.getElementById('menuOverlay');
  var burger  = document.getElementById('hamburger');
  if (overlay) overlay.classList.toggle('open', menuOpen);
  if (burger)  burger.classList.toggle('open', menuOpen);
}}

// ── Navigation with page transition ───────────
function sambhavNav(page) {{
  if (menuOpen) toggleMenu();
  var pt = document.getElementById('pageTransition');
  if (pt) {{ pt.classList.add('active'); }}
  setTimeout(function() {{
    var params = new URLSearchParams(window.location.search);
    params.set('nav', page);
    window.location.search = params.toString();
  }}, 280);
}}

// ── Close menu on Escape ───────────────────────
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape' && menuOpen) toggleMenu();
}});
</script>
"""


def disclaimer_html():
    return """
<div class="disclaimer-bar">
  <span class="disclaimer-txt">Sambhav may be incorrect. Always verify important decisions independently.&nbsp;|&nbsp;Academic use only.</span>
</div>
"""