"use client";

import React, { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';

export default function Page() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle');
  const [job, setJob] = useState(null);
  const [downloadUrls, setDownloadUrls] = useState(null);
  const beforeRef = useRef();
  const afterRef = useRef();
  const wsBefore = useRef(null);
  const wsAfter = useRef(null);

  useEffect(() => {
    return () => {
      if (wsBefore.current) wsBefore.current.destroy();
      if (wsAfter.current) wsAfter.current.destroy();
    };
  }, []);

  const onDrop = useCallback(async (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (!f) return;
    setFile(f);
    await uploadFile(f);
  }, []);

  const uploadFile = async (f) => {
    setStatus('uploading');
    const form = new FormData();
    form.append('file', f);
    form.append('target_lufs', '-10');
    try {
      const res = await axios.post('http://localhost:8000/api/master', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (p) => {
          setStatus(`uploading ${Math.round((p.loaded/p.total)*100)}%`);
        }
      });
      setJob(res.data.job_id);
      setDownloadUrls({ streaming: res.data.streaming_master, highres: res.data.highres_master });
      setStatus('done');

      // load before waveform from local file
      if (wsBefore.current) wsBefore.current.destroy();
      wsBefore.current = WaveSurfer.create({ container: beforeRef.current, waveColor: '#D4A72C', progressColor: '#F2C94C' });
      wsBefore.current.loadBlob(f);

      // load after waveform from the streaming URL
      if (wsAfter.current) wsAfter.current.destroy();
      wsAfter.current = WaveSurfer.create({ container: afterRef.current, waveColor: '#948C7C', progressColor: '#EFE7D8' });
      wsAfter.current.load(`http://localhost:8000${res.data.streaming_master}`);

    } catch (err) {
      console.error(err);
      setStatus('error');
    }
  };

  const onBrowse = (e) => {
    const f = e.target.files[0];
    if (f) uploadFile(f);
  }

  return (
    <div className="min-h-screen bg-[#0E0D0C] text-[#EFE7D8] flex items-center justify-center p-8">
      <div className="w-full max-w-4xl">
        <div onDrop={onDrop} onDragOver={(e)=>e.preventDefault()} className="border-2 border-dashed border-[#3A3226] p-12 rounded-md text-center bg-[#18150F]">
          <h2 className="text-2xl font-semibold">Drop Unmastered Mix Here</h2>
          <p className="text-sm text-[#948C7C] mt-2">or <input type="file" onChange={onBrowse} /></p>
          <div className="mt-6">Status: {status}</div>
        </div>

        <div className="grid grid-cols-2 gap-4 mt-6">
          <div>
            <h3 className="text-sm text-[#948C7C]">Before</h3>
            <div ref={beforeRef} style={{height:120}} className="bg-[#221C14] rounded-md overflow-hidden"></div>
          </div>
          <div>
            <h3 className="text-sm text-[#948C7C]">After (Streaming Master)</h3>
            <div ref={afterRef} style={{height:120}} className="bg-[#221C14] rounded-md overflow-hidden"></div>
          </div>
        </div>

        <div className="mt-6 flex gap-4">
          {downloadUrls && (
            <>
              <a className="px-4 py-2 bg-[#D4A72C] text-black rounded" href={`http://localhost:8000${downloadUrls.streaming}`} target="_blank">Download Streaming Master</a>
              <a className="px-4 py-2 bg-[#948C7C] text-black rounded" href={`http://localhost:8000${downloadUrls.highres}`} target="_blank">Download High-Res Master</a>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
