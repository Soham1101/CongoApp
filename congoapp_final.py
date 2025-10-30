#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: congoapp-final.ipynb
Conversion Date: 2025-10-30T23:13:09.215Z
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Kaggle/Colab-ready: write the HTML app from Python and preview it inline
from IPython.display import HTML, IFrame, display

html_code = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>🦷 Oral Health Helper (Educational Prototype)</title>
<style>
  :root {
    --bg: #f5f6fa;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --brand: #0096c7;
    --brand-dark: #0077b6;
    --border: #dbeafe;
    --primary: #f1bee2;
    --secondary: #4b73b4;
    --accent: #90e0ef;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }

  .container {
    max-width: 980px;
    margin: 30px auto;
    padding: 0 16px;
  }

  header {
    text-align: center;
    margin-bottom: 24px;
  }
  header h1 {
    font-size: 1.9rem;
    color: var(--brand-dark);
    margin-bottom: 4px;
  }
  header p {
    color: var(--muted);
    font-size: 0.95rem;
  }

  /* Tabs */
  .tabs {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }
  .tab-btn {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    cursor: pointer;
    font-weight: 600;
    color: var(--brand-dark);
    transition: all 0.2s ease;
  }
  .tab-btn:hover { background: #e0f2fe; }
  .tab-btn.active {
    background: var(--brand);
    color: white;
    border-color: var(--brand-dark);
  }

  .tab-panel {
    display: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 0 0 12px 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }
  .tab-panel.active { display: block; }

  h2 { color: var(--brand-dark); }

  .btn {
    background: var(--brand);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s ease;
  }
  .btn:hover { background: var(--brand-dark); }
  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    padding: 16px;
    margin-bottom: 16px;
  }

  .stack { display: grid; gap: 14px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

  .preview {
    max-width: 512px;
    width: 100%;
    border: 2px dashed var(--brand);
    border-radius: 12px;
    background: #f0f9ff;
    margin: 10px 0;
  }

  details {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    background: #ffffff;
  }
  details + details { margin-top: 10px; }
  summary {
    font-weight: 600;
    color: var(--brand-dark);
    cursor: pointer;
  }
  small { color: var(--muted); }

  #result h3 { color: var(--brand-dark); margin-bottom: 8px; }

  .notice { color: var(--muted); font-size: 0.9rem; }

  /* Login/Signup */
  .auth-container { max-width: 400px; margin: auto; }
  .auth-container input {
    width: 100%; padding: 10px; margin: 8px 0;
    border: 1px solid #ccc; border-radius: 6px; font-size: 14px;
  }
  .auth-container .message { color: #e63946; font-weight: bold; margin-top: 8px; }
  .auth-container .hidden { display: none; }
  .auth-toggle { text-align: center; margin-top: 10px; }
  .auth-toggle a { cursor: pointer; color: var(--brand-dark); font-weight: bold; text-decoration: none; }
  .auth-toggle a:hover { text-decoration: underline; }

  /* Questionnaire */
  .progress { height: 12px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin-bottom: 25px; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--secondary)); width: 0%; transition: width 0.5s ease; }
  .question { background: #b5f6fa; border-left: 5px solid var(--primary); padding: 15px 20px; margin: 15px 0; border-radius: 10px; transition: all 0.3s ease; }
  .question:hover { transform: scale(1.02); background: #e8f9ff; }
  label { display: block; margin: 8px 0; cursor: pointer; }
  #results { margin-top: 30px; padding: 20px; border-radius: 10px; background: #f0faff; display: none; animation: fadeIn 0.7s ease; }
  .bar { height: 22px; border-radius: 12px; background: #e0e0e0; margin: 10px 0; overflow: hidden; }
  .fill { height: 100%; border-radius: 12px; width: 0%; transition: width 1s ease; }
  .low { background: #48cae4; }
  .moderate { background: #ffb703; }
  .high { background: #ef233c; }
  .condition { font-weight: bold; color: var(--text); }
  @keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🦷 Oral Health Helper</h1>
    <p><strong>Educational prototype — not a medical device.</strong></p>
  </header>

  <nav class="tabs" role="tablist">
    <button class="tab-btn active" data-tab="home">Home</button>
    <button class="tab-btn" data-tab="auth">Login / Signup</button>
    <button class="tab-btn" data-tab="quiz">Self-Assessment</button>
    <button class="tab-btn" data-tab="detect">Upload & Detect</button>
    <button class="tab-btn" data-tab="library">Condition Library</button>
  </nav>

  <!-- Homepage -->
  <section id="home" class="tab-panel active">
    <div class="card">
      <h2>Welcome to the Oral Health Helper</h2>
      <p>This educational prototype helps you learn about oral health, assess risk factors, and explore common dental conditions.</p>
      <ul>
        <li>Track your oral health knowledge</li>
        <li>Answer a short self-assessment questionnaire</li>
        <li>Upload images for prototype analysis</li>
        <li>Learn about dental conditions in our library</li>
      </ul>
      <p class="notice">⚠️ Remember, this tool is for educational purposes only. Consult a dentist for professional advice.</p>
    </div>
  </section>

  <!-- Login / Signup -->
  <section id="auth" class="tab-panel">
    <div class="auth-container">
      <form id="loginForm">
        <input type="email" id="loginEmail" placeholder="Email" required />
        <input type="password" id="loginPassword" placeholder="Password" required />
        <button type="button" class="btn" onclick="login()">Login</button>
        <div id="loginMessage" class="message"></div>
        <div class="auth-toggle">Don't have an account? <a onclick="showSignup()">Sign up</a></div>
      </form>

      <form id="signupForm" class="hidden">
        <input type="text" id="signupName" placeholder="Full Name" required />
        <input type="email" id="signupEmail" placeholder="Email" required />
        <input type="password" id="signupPassword" placeholder="Password" required />
        <input type="password" id="signupConfirm" placeholder="Confirm Password" required />
        <button type="button" class="btn" onclick="signup()">Sign Up</button>
        <div id="signupMessage" class="message"></div>
        <div class="auth-toggle">Already have an account? <a onclick="showLogin()">Login</a></div>
      </form>
    </div>
  </section>

  <!-- Self-Assessment Questionnaire -->
  <section id="quiz" class="tab-panel">
    <div class="stack">
      <div class="progress"><div class="progress-fill" id="progressBar"></div></div>
      <form id="quizForm">
        <div id="questions"></div>
        <button type="button" class="btn" onclick="calculateResults()">Analyze My Results</button>
      </form>
      <div id="results">
        <h2>🩺 Approximate Condition Likelihood</h2>
        <div id="bars"></div>
        <p style="margin-top:10px; color:#555;">⚠️ This tool is for <strong>educational use only</strong>.</p>
      </div>
    </div>
  </section>

  <!-- Upload & Detect -->
  <section id="detect" class="tab-panel">
    <div class="stack">
      <h2>Upload & Detect</h2>
      <input id="fileInput" type="file" accept=".jpg,.jpeg,.png" />
      <small>Upload a clear, well-lit close-up of your teeth or gums.</small>
      <canvas id="previewCanvas" class="preview" aria-label="Image preview"></canvas>
      <div class="row">
        <button id="analyzeBtn" class="btn" disabled>Analyze</button>
        <span id="status" class="notice"></span>
      </div>
      <div id="result" class="card">
        <h3>Result will appear here</h3>
        <div id="result-desc"></div>
      </div>
    </div>
  </section>

  <!-- Condition Library -->
  <section id="library" class="tab-panel">
    <h2>Condition Library</h2>
    <div id="accordion"></div>
  </section>
</div>

<script>
/* TAB LOGIC */
const tabs=document.querySelectorAll(".tab-btn");
tabs.forEach(btn=>btn.addEventListener("click",()=>{
  tabs.forEach(b=>b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));
  btn.classList.add("active");
  document.getElementById(btn.dataset.tab).classList.add("active");
}));

/* CONDITION LIBRARY */
const CONDITIONS={
  "Looks OK": {title:"Looks OK", summary:"No concerning signs detected.", when_to_seek:"See a dentist if issues persist or worsen."},
  "Caries": {title:"Dental Caries (Cavities)", summary:"Decay caused by bacteria and acids damaging enamel.", when_to_seek:"Tooth pain, sensitivity, or visible holes."},
  "Gingivitis": {title:"Gingivitis", summary:"Gum inflammation from plaque buildup.", when_to_seek:"Bleeding gums, redness, or swelling."},
  "Ulcers": {title:"Mouth Ulcers", summary:"Painful sores inside the mouth.", when_to_seek:"Persistent sores or pain over 2 weeks."},
  "Tooth Discoloration": {title:"Tooth Discoloration", summary:"Yellowing or darkening of enamel.", when_to_seek:"Sudden changes or stains that don’t brush off."},
  "Hypodontia": {title:"Hypodontia", summary:"Missing one or more teeth due to developmental issues.", when_to_seek:"Missing teeth not caused by extraction."}
};
const acc=document.getElementById("accordion");
Object.values(CONDITIONS).forEach(c=>{
  const details=document.createElement("details");
  const summary=document.createElement("summary");
  summary.textContent=c.title;
  const body=document.createElement("div");
  body.innerHTML=`<p><b>What it is:</b> ${c.summary}</p><p><b>When to seek care:</b> ${c.when_to_seek}</p>`;
  details.appendChild(summary); details.appendChild(body); acc.appendChild(details);
});

/* LOGIN / SIGNUP */
function showSignup(){document.getElementById("loginForm").classList.add("hidden");document.getElementById("signupForm").classList.remove("hidden");document.getElementById("loginMessage").textContent="";}
function showLogin(){document.getElementById("signupForm").classList.add("hidden");document.getElementById("loginForm").classList.remove("hidden");document.getElementById("signupMessage").textContent="";}
function signup(){
  const name=document.getElementById("signupName").value.trim();
  const email=document.getElementById("signupEmail").value.trim();
  const password=document.getElementById("signupPassword").value;
  const confirm=document.getElementById("signupConfirm").value;
  const message=document.getElementById("signupMessage");
  if(!name||!email||!password||!confirm){message.textContent="⚠️ Please fill out all fields."; return;}
  if(password!==confirm){message.textContent="⚠️ Passwords do not match!"; return;}
  const users=JSON.parse(localStorage.getItem("dentalUsers"))||{};
  if(users[email]){message.textContent="⚠️ Account already exists. Please log in."; return;}
  users[email]={name,password}; localStorage.setItem("dentalUsers",JSON.stringify(users));
  message.style.color="#2a9d8f"; message.textContent="✅ Account created successfully! You can now log in.";
  setTimeout(()=>showLogin(),1500);
}
function login(){
  const email=document.getElementById("loginEmail").value.trim();
  const password=document.getElementById("loginPassword").value;
  const message=document.getElementById("loginMessage");
  const users=JSON.parse(localStorage.getItem("dentalUsers"))||{};
  if(!email||!password){message.textContent="⚠️ Please enter both fields."; return;}
  if(!users[email]){message.textContent="⚠️ No account found. Please sign up first."; return;}
  if(users[email].password!==password){message.textContent="⚠️ Incorrect password."; return;}
  message.style.color="#2a9d8f"; message.textContent="✅ Login successful!";
}

/* QUESTIONNAIRE */
const questions=[
  { text:"Do you notice yellow or brown hard deposits near your gums?", condition:"Calculus" },
  { text:"Do your gums bleed when brushing or flossing?", condition:"Gingivitis" },
  { text:"Do you often have persistent bad breath?", condition:"Gingivitis" },
  { text:"Do you have visible dark spots or holes on your teeth?", condition:"Caries" },
  { text:"Do you feel pain or sensitivity when eating sweets?", condition:"Caries" },
  { text:"Do your gums appear swollen or tender?", condition:"Gingivitis" },
  { text:"Do you have white or yellow ulcers in your mouth?", condition:"Ulcers" },
  { text:"Do these ulcers last more than a week?", condition:"Ulcers" },
  { text:"Are your teeth becoming yellow or darkened?", condition:"Tooth Discoloration" },
  { text:"Do you consume coffee, tea, or tobacco frequently?", condition:"Tooth Discoloration" },
  { text:"Were you born missing one or more teeth?", condition:"Hypodontia" },
  { text:"Are any teeth smaller or more spaced out than others?", condition:"Hypodontia" },
  { text:"Do your teeth feel rough or gritty to the touch?", condition:"Calculus" },
  { text:"Have your gums receded or pulled away from teeth?", condition:"Calculus" },
  { text:"Do you experience sensitivity to hot or cold foods?", condition:"Caries" },
  { text:"Do ulcers recur under stress or after spicy foods?", condition:"Ulcers" },
  { text:"Are there white streaks or uneven coloring on teeth?", condition:"Tooth Discoloration" },
  { text:"Do missing teeth affect your bite or chewing?", condition:"Hypodontia" },
  { text:"Are your gums dark red instead of light pink?", condition:"Gingivitis" },
  { text:"Do you feel buildup on teeth even after brushing?", condition:"Calculus" }
];
function loadQuestions(){
  const container=document.getElementById("questions");
  questions.forEach((q,i)=>{
    container.innerHTML+=`<div class="question"><p><b>Q${i+1}.</b> ${q.text}</p>
      <label><input type="radio" name="q${i}" value="yes" onchange="updateProgress()"> Yes</label>
      <label><input type="radio" name="q${i}" value="no" onchange="updateProgress()"> No</label></div>`
  });
}
loadQuestions();

function updateProgress(){
  const answered=document.querySelectorAll("#quizForm input[type='radio']:checked").length;
  const total=questions.length;
  document.getElementById("progressBar").style.width=`${(answered/total)*100}%`;
}
function calculateResults(){
  const counts={};questions.forEach(q=>counts[q.condition]=0);
  questions.forEach((q,i)=>{
    const val=document.querySelector(`input[name=q${i}]:checked`);
    if(val&&val.value==="yes"){counts[q.condition]+=1;}
  });
  const bars=document.getElementById("bars");bars.innerHTML="";
  Object.keys(counts).forEach(c=>{
    const totalQs=questions.filter(q=>q.condition===c).length;
    const percent=totalQs?Math.round((counts[c]/totalQs)*100):0;
    const div=document.createElement("div");
    div.innerHTML=`<span class="condition">${c}</span>
      <div class="bar"><div class="fill ${percent<35?'low':percent<70?'moderate':'high'}" style="width:${percent}%"></div></div>`;
    bars.appendChild(div);
  });
  document.getElementById("results").style.display="block";
  document.getElementById("results").scrollIntoView({behavior:"smooth"});
}

/* IMAGE PREVIEW + ANALYZE (prototype only) */
const fileInput=document.getElementById("fileInput");
const previewCanvas=document.getElementById("previewCanvas");
const analyzeBtn=document.getElementById("analyzeBtn");
const statusEl=document.getElementById("status");
const resultEl=document.getElementById("result");
const resultDescEl=document.getElementById("result-desc");
let currentBitmap=null;

fileInput.addEventListener("change",async (e)=>{
  const file=e.target.files[0];
  if(!file){analyzeBtn.disabled=true; return;}
  const img=await createImageBitmap(file);
  currentBitmap=img;
  // draw centered square into preview
  const s=Math.min(img.width,img.height);
  const sx=Math.floor((img.width-s)/2);
  const sy=Math.floor((img.height-s)/2);
  previewCanvas.width=512; previewCanvas.height=512;
  const ctx=previewCanvas.getContext("2d");
  ctx.clearRect(0,0,512,512);
  ctx.drawImage(img,sx,sy,s,s,0,0,512,512);
  analyzeBtn.disabled=false;
  statusEl.textContent="";
});

analyzeBtn.addEventListener("click",()=>{
  if(!currentBitmap){return;}
  statusEl.textContent="Analyzing…";
  // downsample to 256x256 and read pixels
  const tmp=document.createElement("canvas");
  tmp.width=256; tmp.height=256;
  const tctx=tmp.getContext("2d");
  tctx.imageSmoothingEnabled=true;
  tctx.drawImage(previewCanvas,0,0,256,256);
  const {data,width,height}=tctx.getImageData(0,0,256,256);
  const prediction=mock_predict(data,width,height);
  const info=CONDITIONS[prediction.label] || CONDITIONS["Looks OK"];
  resultEl.querySelector("h3").innerHTML = `Result: ${info.title} — Confidence: ${prediction.confidence.toFixed(2)}`;
  resultDescEl.innerHTML = `<p><b>What it is:</b> ${info.summary || "Prototype output."}</p>
    <p><b>When to seek care:</b> ${info.when_to_seek || "See a dentist if concerned."}</p>
    <p class="notice">Prototype only — may be inaccurate.</p>`;
  statusEl.textContent="";
});

// Simple heuristic "mock" predictor (educational only)
function mock_predict(rgba,w,h){
  const n=w*h; let sumR=0,sumG=0,sumB=0,sumAll=0,whiteCount=0;
  let graySum=0,graySq=0;
  for(let i=0;i<rgba.length;i+=4){
    const r=rgba[i]/255,g=rgba[i+1]/255,b=rgba[i+2]/255;
    sumR+=r; sumG+=g; sumB+=b;
    const m=(r+g+b)/3; sumAll+=m;
    if(r>0.88&&g>0.88&&b>0.88) whiteCount++;
    graySum+=m; graySq+=m*m;
  }
  const meanR=sumR/n, meanG=sumG/n, meanB=sumB/n;
  const brightness=sumAll/n;
  const redness=meanR-((meanG+meanB)/2);
  const whiteness=whiteCount/n;
  const yellowness=((meanR+meanG)/2)-meanB;
  const grayMean=graySum/n;
  const grayVar=Math.max(0,(graySq/n)-(grayMean*grayMean));

  if(whiteness>0.20 && brightness>0.6) return {label:"Ulcers", confidence:Math.min(0.99,whiteness+0.2)}; // placeholder mapping
  if(redness>0.07 && brightness>0.35) return {label:"Gingivitis", confidence:Math.min(0.95,0.5+redness)};
  if(yellowness>0.1 && brightness<0.65) return {label:"Tooth Discoloration", confidence:Math.min(0.9,0.4+yellowness)};
  if(grayVar>0.035 && brightness<0.55) return {label:"Caries", confidence:0.68};
  return {label:"Looks OK", confidence:0.55};
}
</script>
</body>
</html>
"""

# Write the file
with open("oral-health-helper.html", "w", encoding="utf-8") as f:
    f.write(html_code)

# Show convenient links and an inline preview
display(HTML(
    '<div style="font-family:ui-sans-serif,system-ui">'
    '<a href="oral-health-helper.html" download>⬇️ Download HTML</a> &nbsp;|&nbsp; '
    '<a href="oral-health-helper.html" target="_blank">🌐 Open in new tab</a>'
    "</div>"
))
IFrame(src="oral-health-helper.html", width=900, height=700)
