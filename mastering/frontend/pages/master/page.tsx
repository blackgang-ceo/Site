import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';

// Minimalist, dark-mode one-click mastering page. It posts the uploaded file to the
// FastAPI backend (http://localhost:8000/process) and then displays the streaming/hires
// result links and lets the user A/B the original vs master.

export default function MasterPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState('idle');
  const [results, setResults] = useState<any>(null);
  const originalRef = useRef<HTMLDivElement | null>(null);
  const masterRef = useRef<HTMLDivElement | null>(null);
  const wavesRef = useRef<any>({ orig: null, master: null });

  useEffect(() => {
    return () => {
      // destroy waves on unmount
      if (wavesRef.current.orig) wavesRef.current.orig.destroy();
      if (wavesRef.current.master) wavesRef.current.master.destroy();
    };
  }, []);

  async function upload() {
    if (!file) return;
    setStatus('uploading');
    const fd = new FormData();
    fd.append('file', file);
    // default target lufs
    fd.append('target_lufs', String(-10));
    try {
      const res = await fetch('http://localhost:8000/process', {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) throw new Error('Processing failed');
      const data = await res.json();
      setResults(data);
      setStatus('done');
      // preload waveforms
      loadWaveforms(data);
    } catch (err: any) {
      setStatus('error');
      console.error(err);
      alert('Upload failed: ' + err?.message);
    }
  }

  function loadWaveforms(data: any) {
    const streamingUrl = data.streaming_master ? `http://localhost:8000${data.streaming_master}` : null;
    // original file: create object URL
    const origUrl = file ? URL.createObjectURL(file) : null;

    // destroy existing
    if (wavesRef.current.orig) wavesRef.current.orig.destroy();
    if (wavesRef.current.master) wavesRef.current.master.destroy();

    if (originalRef.current && origUrl) {
      const w = WaveSurfer.create({
        container: originalRef.current,
        waveColor: '#2b2b2b',
        progressColor: '#D4A72C',
        backgroundColor: '#0E0D0C',
        height: 80,
      });
      w.load(origUrl);
      wavesRef.current.orig = w;
    }

    if (masterRef.current && streamingUrl) {
      const w2 = WaveSurfer.create({
        container: masterRef.current,
        waveColor: '#1f1f1f',
        progressColor: '#F2C94C',
        backgroundColor: '#0E0D0C',
        height: 80,
      });
      w2.load(streamingUrl);
      wavesRef.current.master = w2;
    }
  }

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
  }

  return (
    <div style={{background:'#0E0D0C',minHeight:'100vh',color:'#EFE7D8',fontFamily:'Inter, system-ui',padding:'28px'}}>
      <h1 style={{fontSize:22,fontWeight:700,letterSpacing:'0.03em'}}>BLACKGANG One-Click Mastering</h1>
      <p style={{color:'#948C7C',marginTop:6}}>Drag & drop your mix (wav/mp3) — the engine will analyze and produce streaming + hi-res masters.</p>

      <div style={{marginTop:18,display:'flex',gap:12}}>
        <div style={{flex:1}}>
          <label style={{display:'block',border:'1px dashed #3A3226',padding:26,borderRadius:6,background:'#18150F'}}>
            <input type="file" accept="audio/*" onChange={onFile} style={{display:'none'}} />
            <div style={{textAlign:'center'}}>
              <div style={{fontWeight:600,fontSize:16}}>Drop Unmastered Mix Here</div>
              <div style={{color:'#948C7C',marginTop:6}}>WAV / MP3 / AIFF / M4A</div>
            </div>
          </label>
          <div style={{marginTop:12,display:'flex',gap:8}}>
            <button onClick={upload} disabled={!file || status==='uploading'} style={{background:'#D4A72C',border:'none',padding:'10px 14px',fontWeight:700,borderRadius:4,color:'#0E0D0C'}}>Master</button>
            <div style={{color:'#948C7C',alignSelf:'center'}}>{status}</div>
          </div>
        </div>

        <div style={{width:320,background:'#18150F',padding:12,borderRadius:6,border:'1px solid #3A3226'}}>
          <div style={{fontSize:12,color:'#948C7C'}}>Progress</div>
          <div style={{marginTop:8}}>
            <div style={{height:8,background:'#221C14',borderRadius:4}}>
              <div style={{width: status==='uploading' ? '60%' : status==='done' ? '100%' : '6%', height:'100%', background:'#D4A72C', borderRadius:4}} />
            </div>
            <div style={{fontSize:12,color:'#948C7C',marginTop:8}}>{status==='idle'?'Waiting':'Processing'}</div>
          </div>
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:18,marginTop:22}}>
        <div style={{background:'#18150F',padding:10,borderRadius:6,border:'1px solid #3A3226'}}>
          <div style={{fontSize:12,color:'#948C7C',marginBottom:8}}>Original</div>
          <div ref={originalRef}></div>
        </div>
        <div style={{background:'#18150F',padding:10,borderRadius:6,border:'1px solid #3A3226'}}>
          <div style={{fontSize:12,color:'#948C7C',marginBottom:8}}>Master (streaming)</div>
          <div ref={masterRef}></div>
        </div>
      </div>

      {results && (
        <div style={{marginTop:20}}>
          <h3 style={{margin:0}}>Results</h3>
          <pre style={{background:'#0E0D0C',border:'1px solid #3A3226',padding:12,borderRadius:6,color:'#EFE7D8'}}>{JSON.stringify(results.analysis, null, 2)}</pre>
          <div style={{marginTop:8,display:'flex',gap:8}}>
            <a href={`http://localhost:8000${results.streaming_master}`} download style={{background:'#D4A72C',padding:'8px 12px',fontWeight:700,borderRadius:4,color:'#0E0D0C',textDecoration:'none'}}>Download Streaming Master</a>
            <a href={`http://localhost:8000${results.hires_master}`} download style={{background:'#F2C94C',padding:'8px 12px',fontWeight:700,borderRadius:4,color:'#0E0D0C',textDecoration:'none'}}>Download Hi-Res Master</a>
          </div>
        </div>
      )}

    </div>
  );
}
